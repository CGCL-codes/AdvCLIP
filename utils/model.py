import torch
import pickle
import math
import clip
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from transformers import BertConfig, BertTokenizer, BertModel
from torchvision.models import alexnet, resnet18, resnet50, inception_v3, vgg19
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

seed = 1
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


class Generator224(torch.nn.Module):
    def __init__(self, input_dim, num_filters, output_dim, batch_size):
        super(Generator224, self).__init__()

        # Hidden layers
        self.hidden_layer = torch.nn.Sequential()
        for i in range(len(num_filters)):
            # Deconvolutional layer
            if i == 0:
                deconv = torch.nn.ConvTranspose2d(input_dim, num_filters[i], kernel_size=4, stride=1, padding=0)
            else:
                deconv = torch.nn.ConvTranspose2d(num_filters[i-1], num_filters[i], kernel_size=4, stride=2, padding=1)

            deconv_name = 'deconv' + str(i + 1)
            self.hidden_layer.add_module(deconv_name, deconv)

            # Initializer
            torch.nn.init.normal(deconv.weight, mean=0.0, std=0.02)
            torch.nn.init.constant(deconv.bias, 0.0)

            # Batch normalization
            bn_name = 'bn' + str(i + 1)
            self.hidden_layer.add_module(bn_name, torch.nn.BatchNorm2d(num_filters[i]))

            # Activation
            act_name = 'act' + str(i + 1)
            self.hidden_layer.add_module(act_name, torch.nn.ReLU())


        # Output layer
        self.output_layer = torch.nn.Sequential()
        # Deconvolutional layer
        out = torch.nn.ConvTranspose2d(num_filters[i], output_dim, kernel_size=4, stride=8, padding=14)
        self.output_layer.add_module('out', out)
        # Initializer
        torch.nn.init.normal(out.weight, mean=0.0, std=0.02)
        torch.nn.init.constant(out.bias, 0.0)
        # Activation
        self.output_layer.add_module('act', torch.nn.Tanh())

        # Residual layer
        self.residual_layer = torch.nn.Sequential()
        res = torch.nn.Conv2d(in_channels=batch_size,out_channels=1,kernel_size=3,stride=1,padding=1)
        self.residual_layer.add_module('res', res)
        torch.nn.init.normal(res.weight, mean=0.0, std=0.02)
        torch.nn.init.constant(res.bias, 0.0)

    def forward(self, x):
        h = self.hidden_layer(x)
        out = self.output_layer(h)
        out_ = out.permute(1,0,2,3)
        adv_out = self.residual_layer(out_)
        adv_out_ = adv_out.permute(1,0,2,3)
        return adv_out_

class Discriminator224(torch.nn.Module):
    def __init__(self, input_dim, num_filters, output_dim):
        super(Discriminator224, self).__init__()

        # Hidden layers
        self.hidden_layer = torch.nn.Sequential()
        for i in range(len(num_filters)):
            # Convolutional layer
            if i == 0:
                conv = torch.nn.Conv2d(input_dim, num_filters[i], kernel_size=4, stride=2, padding=1)
            else:
                conv = torch.nn.Conv2d(num_filters[i-1], num_filters[i], kernel_size=4, stride=2, padding=1)

            conv_name = 'conv' + str(i + 1)
            self.hidden_layer.add_module(conv_name, conv)

            # Initializer
            torch.nn.init.normal(conv.weight, mean=0.0, std=0.02)
            torch.nn.init.constant(conv.bias, 0.0)

            # Batch normalization
            if i != 0:
                bn_name = 'bn' + str(i + 1)
                self.hidden_layer.add_module(bn_name, torch.nn.BatchNorm2d(num_filters[i]))

            # Activation
            act_name = 'act' + str(i + 1)
            self.hidden_layer.add_module(act_name, torch.nn.LeakyReLU(0.2))

        # Output layer
        self.output_layer = torch.nn.Sequential()
        # Convolutional layer
        out = torch.nn.Conv2d(num_filters[i], output_dim, kernel_size=14, stride=2, padding=0)

        self.output_layer.add_module('out', out)
        # Initializer
        torch.nn.init.normal(out.weight, mean=0.0, std=0.02)
        torch.nn.init.constant(out.bias, 0.0)

        # Activation
        self.output_layer.add_module('act', torch.nn.Sigmoid())


    def forward(self, x):
        h = self.hidden_layer(x)
        out = self.output_layer(h)

        return out


class NonLinearClassifier(nn.Module):
    """Linear classifier"""
    def __init__(self, feat_dim=512, num_classes=10):
        super(NonLinearClassifier, self).__init__()
        self.fc1 = nn.Linear(feat_dim, feat_dim)
        self.dropout = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(feat_dim, feat_dim)
        self.fc3 = nn.Linear(feat_dim, num_classes)

    def forward(self, features):
        features = F.relu(self.fc1(features))
        features = F.relu(self.fc2(self.dropout(features)))
        return self.fc3(features)



def l1norm(X, dim, eps=1e-8):
    """L1-normalize columns of X
    """
    norm = torch.abs(X).sum(dim=dim, keepdim=True) + eps
    X = torch.div(X, norm)
    return X


def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X


def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class ImgNN(nn.Module):
    """Network to learn image representations"""
    def __init__(self, input_dim=2048, mindum_dim=512, out_dim=128, dropout_prob=0.1):
        super(ImgNN, self).__init__()
        self.denseL1 = nn.Linear(input_dim, mindum_dim)
        self.denseL2 = nn.Linear(mindum_dim, out_dim)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        # out = F.relu(self.denseL1(x))
        out = gelu(self.denseL1(x))
        out = self.dropout(self.denseL2(out))
        return out


class TextNN(nn.Module):
    """Network to learn text representations"""
    def __init__(self, input_dim=768, mindum_dim=512, out_dim=128, dropout_prob=0.1):
        super(TextNN, self).__init__()
        self.denseL1 = nn.Linear(input_dim, mindum_dim)
        self.denseL2 = nn.Linear(mindum_dim, out_dim)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        # out = F.relu(self.denseL1(x))
        out = gelu(self.denseL1(x))
        out = self.dropout(self.denseL2(out))
        return out


class model(nn.Module):
    # def __init__(self, num_class, img_dim=1024, text_dim=1024, mid_dim=256, feature_dim=1024, init_weight=True):
    def __init__(self, num_class, img_dim, text_dim, mid_dim, feature_dim, init_weight=True):

        super(model, self).__init__()

        self.imgnn = ImgNN(input_dim=img_dim, mindum_dim=mid_dim, out_dim=feature_dim)
        self.textnn = TextNN(input_dim=text_dim, mindum_dim=mid_dim, out_dim=feature_dim)

        self.n_classes = num_class
        self.feat_dim = feature_dim
        self.predictLayer = nn.Linear(self.feat_dim, self.n_classes, bias=True)  # 不考虑bias，权重归一化之后就是Proxy-NCA, Normlized Softmax
        self.centers = nn.Parameter(torch.randn(self.feat_dim, self.n_classes), requires_grad=False)
        if init_weight:
            self.__init_weight()

    def __init_weight(self):
        nn.init.kaiming_normal_(self.centers, mode='fan_out')
        nn.init.kaiming_normal_(self.predictLayer.weight.data, mode='fan_out')

    def forward(self, img, text):
        # the normalization of class proxies, the more obvious its effect is in the higher-dimensional representation space
        self.predictLayer.weight.data = l2norm(self.predictLayer.weight.data, dim=-1)
        self.centers.data = l2norm(self.centers.data, dim=0)

        img_features = self.imgnn(img.float())
        img_features = l2norm(img_features, dim=1)
        img_pred = self.predictLayer(img_features)

        text_features = self.textnn(text.float())
        text_features = l2norm(text_features, dim=1)
        text_pred = self.predictLayer(text_features)

        return self.centers, img_features, text_features, img_pred, text_pred


class fin_model(nn.Module):
    # def __init__(self, num_class, img_dim=1024, text_dim=1024, mid_dim=256, feature_dim=1024, init_weight=True):
    def __init__(self, clip, num_class, img_dim, text_dim, mid_dim, feature_dim, init_weight=True):

        super(fin_model, self).__init__()

        self.imgnn = ImgNN(input_dim=img_dim, mindum_dim=mid_dim, out_dim=feature_dim)
        self.textnn = TextNN(input_dim=text_dim, mindum_dim=mid_dim, out_dim=feature_dim)
        self.clip = clip
        self.n_classes = num_class
        self.feat_dim = feature_dim
        self.predictLayer = nn.Linear(self.feat_dim, self.n_classes, bias=True)  # 不考虑bias，权重归一化之后就是Proxy-NCA, Normlized Softmax
        self.centers = nn.Parameter(torch.randn(self.feat_dim, self.n_classes), requires_grad=False)
        if init_weight:
            self.__init_weight()

    def __init_weight(self):
        nn.init.kaiming_normal_(self.centers, mode='fan_out')
        nn.init.kaiming_normal_(self.predictLayer.weight.data, mode='fan_out')

    def forward(self, imgs, texts):
        # the normalization of class proxies, the more obvious its effect is in the higher-dimensional representation space
        self.predictLayer.weight.data = l2norm(self.predictLayer.weight.data, dim=-1)
        self.centers.data = l2norm(self.centers.data, dim=0)
        img = self.clip.encode_image(imgs.squeeze())
        text = self.clip.encode_text(texts.squeeze())

        img_features = self.imgnn(img.float())
        img_features = l2norm(img_features, dim=1)
        img_pred = self.predictLayer(img_features)

        text_features = self.textnn(text.float())
        text_features = l2norm(text_features, dim=1)
        text_pred = self.predictLayer(text_features)

        return self.centers, img_features, text_features, img_pred, text_pred
