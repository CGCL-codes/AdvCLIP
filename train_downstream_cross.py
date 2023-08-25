import os
import copy
import torch
import time
import numpy as np
import clip
import pickle
import argparse
import torch.optim as optim
from utils.model import model
from utils.evaluate import fx_calc_map_label
from utils.metrics import Contrastive_Loss
from utils.load_data import load_dataset

def load_config():
    parser = argparse.ArgumentParser(description='Cross downstream task')
    parser.add_argument('--seed', default=100, type=int, help='which seed the code runs on')
    parser.add_argument('--dataset', type=str, default='xmedianet', choices=['nus-wide', 'pascal', 'wikipedia', 'xmedianet'])
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--save', type=bool, default=False)
    parser.add_argument('--device', default="cuda:1", type=str, help='which gpu the code runs on')
    parser.add_argument('--victim', default='ViT-B/32', choices=['ViT-L/14', 'ViT-B/16', 'ViT-B/32', 'RN50', 'RN101'])
    parser.add_argument('--num_epochs', type=int, default=500)
    args = parser.parse_args()
    return args

def train(clip_model, model, loader, optimizer, num_class):
    model.train()
    running_loss = 0.0
    for img, text, labels, id in loader:
        optimizer.zero_grad()
        text = text.to(device)
        img = img.to(device)
        img = clip_model.encode_image(img.squeeze())
        text = clip_model.encode_text(text.squeeze())
        label_realvalue = labels.int().type(torch.long).to(device)
        centers, img_feature, text_feature, img_predict, text_predict = model(img, text)
        # centers = centers[:img_feature.shape[1]]  # multiple GPUs
        loss = Contrastive_Loss(img_feature, text_feature, label_realvalue)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    return running_loss / len(loader)


def evaluate(clip_model, model, loader):
    model.eval()
    running_loss = 0.0
    t_imgs, t_txts, t_labels = [], [], []
    with torch.no_grad():
        for img, text, labels, id in loader:
            text = text.to(device)
            img = img.to(device)

            img = clip_model.encode_image(img.squeeze())
            text = clip_model.encode_text(text.squeeze())

            labels = labels.int().to(device)
            _, img_feature, text_feature, img_predict, text_predict = model(img, text)

            t_imgs.append(img_feature.cpu().numpy())
            t_txts.append(text_feature.cpu().numpy())
            t_labels.append(labels.cpu().numpy())

    t_imgs = np.concatenate(t_imgs)  # for visualization
    t_txts = np.concatenate(t_txts)  # for visualization
    t_labels = np.concatenate(t_labels)
    i_map = fx_calc_map_label(t_imgs, t_txts, t_labels)
    t_map = fx_calc_map_label(t_txts, t_imgs, t_labels)
    print('Image to Text: MAP: {:.4f}'.format(i_map))
    print('Text to Image: MAP: {:.4f}'.format(t_map))

    return i_map, t_map, t_imgs, t_txts, t_labels


if __name__ == '__main__':

    args = load_config()
    # init the random seeds
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    dataloaders = load_dataset(args.dataset, args.batch_size)
    train_loader = dataloaders['train']
    test_loader = dataloaders['test']
    print(len(train_loader), len(test_loader))

    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    USE_CUDA = torch.cuda.is_available()
    device = torch.device("cuda:1" if USE_CUDA else "cpu")

    if args.dataset == 'wikipedia':
        num_class = 10
    elif args.dataset == 'pascal':
        num_class = 20
    elif args.dataset == 'xmedianet':
        num_class = 100
    elif args.dataset == 'nus-wide':
        num_class = 81

    if args.victim == 'ViT-L/14':
        img_dim = 768
        text_dim = 768
        feature_dim = 768
    elif args.victim == 'RN50':
        img_dim = 1024
        text_dim = 1024
        feature_dim = 1024
    elif args.victim == 'ViT-B/16' or args.victim == 'ViT-B/32' or args.victim == 'RN101':
        img_dim = 512
        text_dim = 512
        feature_dim = 512
    else:
        img_dim = 512
        text_dim = 512
        feature_dim = 512


    clip_model, preprocess = clip.load(args.victim, device=device)

    MAX_EPOCH = args.num_epochs
    temperature = 1.0
    lr = 1e-4
    betas = (0.5, 0.999)
    weight_decay = 0
    early_stop = 10

    model_ft = model(num_class=num_class, img_dim=img_dim, text_dim=text_dim, mid_dim=256, feature_dim=feature_dim).to(device)
    model_ft.to(device)

    params_to_update = list(model_ft.parameters())
    total = sum([param.nelement() for param in params_to_update])
    print("Number of parameter: %.2fM" % (total / 1e6))
    # Observe that all parameters are being optimized
    optimizer_all = optim.Adam(params_to_update, lr=lr, betas=betas)
    for state in [1]:
        print('...Training is beginning...', state)
        # Train and evaluate
        train_loss_history = []
        test_loss_history = []
        i_map = []
        t_map = []
        best_map = 0.0
        no_up = 0  # early stop
        best_model_wts = copy.deepcopy(model_ft.state_dict())

        for epoch in range(MAX_EPOCH):
            print('==============================')
            start_time = time.time()
            train_loss = train(clip_model, model_ft, train_loader, optimizer_all, num_class=num_class)
            print('Train loss: ', train_loss)

            img2text, text2img, t_imgs, t_txts, t_labels = evaluate(clip_model, model_ft, test_loader)
            i_map.append(img2text)
            t_map.append(text2img)

            time_elapsed = time.time() - start_time
            print(f'Epoch: {epoch + 1:02} | Epoch Time: {int(time_elapsed // 60)}m {int(time_elapsed % 60)}s')

            if (img2text + text2img) / 2. > best_map:
                best_map = (img2text + text2img) / 2.
                print('New Best model')
                no_up = 0
                best_model_wts = copy.deepcopy(model_ft.state_dict())

                if args.save == True:
                    # Logging
                    model_save_path = os.path.join('output', 'model', str(args.victim), str(args.dataset))
                    if not os.path.exists(model_save_path):
                        os.makedirs(model_save_path)
                    torch.save(model_ft.state_dict(), '{}/{}'.format(model_save_path, str(args.dataset) + '.' + 'pt'))

                    np.savez('{}/{}.npz'.format(model_save_path, best_map), image=t_imgs, text=t_txts, label=t_labels)
            else:
                no_up += 1
            if no_up >= early_stop:
                break
        print('==============================')
        print(f'Best average mAP: {best_map:.4f}, Epoch: {epoch + 1 - early_stop}')
