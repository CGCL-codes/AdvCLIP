import os
import torch
import copy
import time
import clip
import pickle
import argparse
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import torch.optim as optim
from torch.utils.data import DataLoader
from utils.load_data import load_dataset
from utils.model import NonLinearClassifier

def load_config():
    parser = argparse.ArgumentParser(description='Solo downstream task')
    parser.add_argument('--seed', default=100, type=int, help='which seed the code runs on')
    parser.add_argument('--dataset', type=str, default='xmedianet', choices=['nus-wide', 'pascal', 'wikipedia', 'xmedianet'])
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--save', type=bool, default=True)
    parser.add_argument('--device', default="cuda:1", type=str, help='which gpu the code runs on')
    parser.add_argument('--victim', default='ViT-L/14', choices=['ViT-L/14', 'ViT-B/16', 'ViT-B/32', 'RN50', 'RN101'])
    parser.add_argument('--epochs', type=int, default=20)
    args = parser.parse_args()
    return args

def accuracy(output, target, topk=(1,)):
    with torch.no_grad():

        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)

        pred = pred.t()

        correct = pred.eq(target.view(1, -1).expand_as(pred))
        # correct = pred.eq(target.expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def test(encoder, classifier, test_loader, device):
    top1_accuracy = 0
    top5_accuracy = 0
    encoder.eval()
    classifier.eval()

    with torch.no_grad():
        for counter, (x_batch, text_batch, y_batch, id) in enumerate(tqdm(test_loader)):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            h = encoder.encode_image(x_batch.squeeze())
            x_in = h.view(h.size(0), -1)
            # x_in = torch.tensor(x_in, dtype=torch.float)
            logits = classifier(x_in.float())
            top1, top5 = accuracy(logits, y_batch, topk=(1, 5))
            top1_accuracy += top1[0]
            top5_accuracy += top5[0]

        top1_accuracy /= (counter + 1)
        top5_accuracy /= (counter + 1)

    return top1_accuracy.item(), top5_accuracy.item()


def classify(args, encoder, train_loader, test_loader, num_class, feat_dim, device):

    F = NonLinearClassifier(feat_dim=feat_dim, num_classes=num_class)
    F.to(device)
    encoder.to(device)
    # classifier
    my_optimizer = torch.optim.Adam(F.parameters(), lr=0.005, weight_decay=0.0008)
    my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=my_optimizer, gamma=0.96)
    criterion = torch.nn.CrossEntropyLoss().to(device)

    model_save_path = os.path.join('output', 'solo_model', str(args.victim), str(args.dataset))
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)

    F.train()
    encoder.eval()

    for epoch in range(args.epochs):
        start = time.time()
        top1_train_accuracy = 0
        for counter, (x_batch, text_batch, y_batch, id) in enumerate(tqdm(train_loader)):
            my_optimizer.zero_grad()
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            h = encoder.encode_image(x_batch.squeeze())
            downstream_input = h.view(h.size(0), -1)
            logits = F(downstream_input.float())
            loss = criterion(logits, y_batch)
            top1 = accuracy(logits, y_batch, topk=(1,))
            top1_train_accuracy += top1[0]
            loss.backward()
            my_optimizer.step()

        end = time.time()
        F.train()
        clean_acc_t1, clean_acc_t5 = test(encoder, F, test_loader, device)
        if args.save == True:
            torch.save(F.state_dict(), '{}/{}'.format(model_save_path, str(victim_name) + '_' + str(args.dataset)  + '_' + str(
                round(clean_acc_t1, 4)) + '_' + str(epoch + 1) + '.pth'))
        my_lr_scheduler.step()
        top1_train_accuracy /= (counter + 1)
        print('Epoch [%d/%d], Top1 train acc: %.4f, Top1 test acc: %.4f, Time: %.4f'
              % (epoch + 1, args.epochs, top1_train_accuracy.item(), clean_acc_t1, (end - start)))

    return clean_acc_t1, clean_acc_t5


def main():
    args = load_config()
    # Set the seed and determine the GPU
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    # set device for CLIP
    USE_CUDA = torch.cuda.is_available()
    device = torch.device(args.device if USE_CUDA else "cpu")
    dataloaders = load_dataset(args.dataset, args.batch_size)
    train_loader = dataloaders['train']
    test_loader = dataloaders['test']
    print(len(train_loader), len(test_loader))

    if args.dataset == 'wikipedia':
        num_class = 10
    elif args.dataset == 'pascal':
        num_class = 21
    elif args.dataset == 'xmedianet':
        num_class = 100
    elif args.dataset == 'nus-wide':
        num_class = 81

    if args.victim == 'ViT-L/14':
        feat_dim = 768
    elif args.victim == 'RN50':
        feat_dim = 1024
    elif args.victim == 'ViT-B/16' or args.victim == 'ViT-B/32' or args.victim == 'RN101':
        feat_dim = 512
    else:
        feat_dim = 512

    clip_model, preprocess = clip.load(args.victim, device=device)  # ['RN50', 'RN101', 'RN50x4', 'RN50x16', 'ViT-B/32', 'ViT-B/16']
    
    clean_acc_t1, clean_acc_t5 = classify(args, clip_model, train_loader, test_loader, num_class, feat_dim, device)
    print('Clean downstream accuracy: %.4f%%'% (clean_acc_t1))


if __name__ == "__main__":
    main()
