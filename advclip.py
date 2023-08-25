import os
import csv
import torch
import clip
import json
import argparse
import datetime
import torch.nn as nn
from torch.autograd import Variable
from utils.load_data import load_dataset
from utils.evaluate import evaluate, adv_evaluate, adv_eval
from utils.model import Generator224, Discriminator224
from utils.nce import InfoNCE
from utils.metrics import umap, supervised_umap
from utils.patch_utils import patch_initialization, mask_generation, clamp_patch

def load_config():
    parser = argparse.ArgumentParser(description='Train AdvCLIP')
    parser.add_argument('--seed', default=100, type=int, help='which seed the code runs on')
    parser.add_argument('--dataset', type=str, default='nus-wide', choices=['nus-wide', 'pascal', 'wikipedia', 'xmedianet'])
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--save', type=bool, default=True)
    parser.add_argument('--mode', default="gan_patch", type=str)
    parser.add_argument('--device', default="cuda:1", type=str, help='which gpu the code runs on')
    parser.add_argument('--victim', default='ViT-B/16', choices=['ViT-L/14', 'ViT-B/16', 'ViT-B/32', 'RN50', 'RN101'])
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--alpha', type=int, default=10)
    parser.add_argument('--beta', type=int, default=5)
    parser.add_argument('--gamma', type=int, default=5)
    parser.add_argument('--delta', type=int, default=1)
    parser.add_argument('--noise_percentage', type=float, default=0.03)
    args = parser.parse_args()
    return args

def advclip(args, train_loader, test_loader, model, mask, device):
    # init the GAN
    G_input_dim = 100
    G_output_dim = 3

    D_input_dim = 3
    D_output_dim = 1

    num_filters = [1024, 512, 256, 128]
    learning_rate = 0.0002
    betas = (0.5, 0.999)

    G = Generator224(G_input_dim, num_filters, G_output_dim, args.batch_size)
    D = Discriminator224(D_input_dim, num_filters[::-1], D_output_dim)

    model.eval()
    G.to(device)
    D.to(device)

    # criterion_l2
    criterion_l2 = torch.nn.MSELoss()
    criterion_contrastive = InfoNCE()
    criterion_bce = torch.nn.BCELoss()

    # Optimizers
    G_optimizer = torch.optim.Adam(G.parameters(), lr=learning_rate, betas=betas)
    D_optimizer = torch.optim.Adam(D.parameters(), lr=learning_rate, betas=betas)

    # Training GAN
    # define a global fix noise z
    z = torch.randn(args.batch_size, G_input_dim).view(-1, G_input_dim, 1, 1)
    z = Variable(z.to(device))

    for epoch in range(args.num_epochs):

        for i, (img, text, labels, id) in enumerate(train_loader):
            mini_batch = img.size()[0]
            x = Variable(img.squeeze().to(device))
            new_shape = x.shape
            ##########################################  D  ####################################################

            y_real_ = Variable(torch.ones(mini_batch).to(device))
            y_fake_ = Variable(torch.zeros(mini_batch).to(device))
            # Train discriminator with real data

            D_real_decision = D(x).squeeze()

            D_real_loss = criterion_bce(D_real_decision, y_real_)
            uap_noise = G(z).squeeze()
            uap_noise = clamp_patch(args, uap_noise)
            uap_noise.to(device)

            # add the uap
            f_x = torch.mul(mask.type(torch.FloatTensor),
                             uap_noise.type(torch.FloatTensor)) + torch.mul(
                1 - mask.expand(new_shape).type(torch.FloatTensor), x.type(torch.FloatTensor))

            f_x.to(device)

            D_fake_decision = D(f_x.to(device)).squeeze()
            D_fake_loss = criterion_bce(D_fake_decision, y_fake_)

            # Back propagation
            D_loss = D_real_loss + D_fake_loss

            D.zero_grad()
            D_loss.backward()
            D_optimizer.step()

            for p in D.parameters():
                p.data.clamp_(-0.01, 0.01)

            ##########################################  G  ####################################################

            uap_noise = G(z).squeeze()
            uap_noise = clamp_patch(args, uap_noise)

            uap_noise.to(device)

            # add the uap
            f_x = torch.mul(mask.type(torch.FloatTensor),
                             uap_noise.type(torch.FloatTensor)) + torch.mul(
                1 - mask.expand(new_shape).type(torch.FloatTensor), x.type(torch.FloatTensor))

            f_x.to(device)

            # l_{2} loss
            reconstruction_loss = criterion_l2(f_x.to(device), x.to(device))

            clean_output = model.encode_image(x.to(device))
            clean_output_text = model.encode_text(text.squeeze().to(device))
            per_output = model.encode_image(f_x.to(device))

            # GAN loss
            D_fake_decision = D(f_x.to(device)).squeeze()
            GAN_loss = criterion_bce(D_fake_decision, y_real_)

            adv_loss_pos1 = criterion_contrastive(clean_output, per_output).mean()
            adv_loss_pos2 = criterion_contrastive(per_output, clean_output_text).mean()
            adv_loss1 = -adv_loss_pos1
            adv_loss2 = -adv_loss_pos2

            adv_loss = adv_loss1 + args.beta * adv_loss2

            umap_loss_pos1 = - umap(clean_output, per_output)
            umap_loss_pos2 = - umap(per_output, clean_output_text)
            umap_loss = umap_loss_pos1 + args.gamma * umap_loss_pos2

            G_loss = GAN_loss + args.alpha * adv_loss + reconstruction_loss + args.delta * umap_loss

            # Back propagation
            D.zero_grad()
            G.zero_grad()
            G_loss.backward()
            G_optimizer.step()
            if i % 1 == 0:
                print('Epoch [%d/%d], Step [%d/%d], Adv_loss: %.4f, L2_loss: %.4f, G_loss: %.4f, D_loss: %.4f'
                    % (epoch + 1, args.num_epochs, i + 1, len(train_loader), adv_loss.item(), reconstruction_loss.item(), G_loss.item(), D_loss.item()))

        clean_acc, p_acc, fr = adv_eval(model, train_loader, test_loader, uap_noise, mask, device)
        i_map, t_map, t_imgs, t_txts, t_labels = evaluate(model, test_loader, device)
        p_i_map, p_t_map, p_t_imgs, p_t_txts, p_t_labels = adv_evaluate(model, test_loader, uap_noise, mask, device)

        if args.save == True:
            # save uap result
            uap_save_path = os.path.join('output', 'uap', str(args.mode), str(args.victim_name), str(args.dataset), str(args.noise_percentage))
            if not os.path.exists(uap_save_path):
                os.makedirs(uap_save_path)

            generator_save_path = os.path.join('output', 'uap', str(args.mode), str(args.victim_name), str(args.dataset),
                                         str(args.noise_percentage), 'generator')
            if not os.path.exists(generator_save_path):
                os.makedirs(generator_save_path)

            discriminator_save_path = os.path.join('output', 'uap', str(args.mode), str(args.victim_name), str(args.dataset),
                                               str(args.noise_percentage), 'discriminator')
            if not os.path.exists(discriminator_save_path):
                os.makedirs(discriminator_save_path)

            torch.save(uap_noise.cpu().data, '{}/{}'.format(uap_save_path, 'uap_gan_' + str(round(((((i_map - p_i_map)/i_map) * 100 + ((t_map - p_t_map)/t_map)*100) / 2), 2)) + '_' + str(epoch + 1) + '.pt'))
            torch.save(G.state_dict(), '{}/{}'.format(generator_save_path, str(args.victim_name) + '_' + str(args.dataset)  + '_' + str(epoch + 1) + '.pth'))
            torch.save(D.state_dict(), '{}/{}'.format(discriminator_save_path, str(args.victim_name) + '_' + str(args.dataset)  + '_' + str(epoch + 1) + '.pth'))

        print("####################################### Attack #######################################")
        print('Classifiy results: Clean_Acc: {:.4f}, Per_Acc: {:.4f}, Fooling Rate: {:.4f} '.format(round(clean_acc, 4), round(p_acc, 4), round(fr, 4)))
        print('Retrieval results: Clean_I-T {:.4f}, Clean_T-I {:.4f}, Adv_I-T {:.4f}, Adv_T-I {:.4f}, '.format(round(i_map, 4), round(t_map, 4),round(p_i_map, 4), round(p_t_map, 4)))
        print('Attack Performance: Decline_I-T {:.4f}%, Decline_T-I {:.4f}%, '.format(round(((i_map - p_i_map)/i_map)*100, 4), round(((t_map - p_t_map)/t_map)*100, 4)))
        print("#######################################  End  #######################################")

    return clean_acc, p_acc, fr, i_map, t_map, p_i_map, p_t_map

if __name__ == '__main__':
    args = load_config()
    # init random seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    # set device for CLIP

    USE_CUDA = torch.cuda.is_available()
    device = torch.device(args.device if USE_CUDA else "cpu")

    now_time = datetime.datetime.now().strftime('%Y_%m_%d')

    if args.victim == 'RN50':
        args.victim_name = 'RN50'
    elif args.victim == 'RN101':
        args.victim_name = 'RN101'
    elif args.victim == 'ViT-B/16':
        args.victim_name = 'ViT-B16'
    elif args.victim == 'ViT-B/32':
        args.victim_name = 'ViT-B32'
    elif args.victim == 'ViT-L/14':
        args.victim_name = 'ViT-L14'

    # load cross-modal dataset
    dataloaders = load_dataset(args.dataset, args.batch_size)
    train_loader = dataloaders['train']
    test_loader = dataloaders['test']
    # print dataset length
    print("now_time:", now_time, "train_loader：", len(train_loader), "test_loader：", len(test_loader))
    # load pre-trainede CLIP model
    model, preprocess = clip.load(args.victim, device=device)
    # init patch
    patch = patch_initialization(args)
    mask, applied_patch, x, y = mask_generation(args, patch)
    applied_patch = torch.from_numpy(applied_patch)
    mask = torch.from_numpy(mask)

    clean_acc, p_acc, fr, i_map, t_map, p_i_map, p_t_map = advclip(args, train_loader, test_loader, model, mask, device)

    # save results
    if args.save == True:
        # Logging
        log_save_path = os.path.join('output', 'log', str(args.mode), str(args.victim_name),
                                     str(args.dataset), str(args.noise_percentage))
        if not os.path.exists(log_save_path):
            os.makedirs(log_save_path)

        # Dump args
        with open(log_save_path + '/args.json', 'w') as fid:
            json.dump(args.__dict__, fid, indent=2)

        final_log_save_path = os.path.join('output', 'log', str(args.mode))
        if not os.path.exists(final_log_save_path):
            os.makedirs(final_log_save_path)
        final_result = []
        final_result_ = {"now_time": now_time,
                         "victim": str(args.victim),
                         "dataset": str(args.dataset),
                         "noise_percentage": str(round(args.noise_percentage, 4)),
                         "i_map": round(i_map*100, 4),
                         "t_map": round(t_map*100, 4),
                         "fooling_rate": round(fr, 4),
                         "p_i_map": round(p_i_map*100, 4),
                         "p_t_map": round(p_t_map*100, 4),
                         "d_i_t": round(((i_map - p_i_map)/i_map)*100, 4),
                         "d_t_i": round(((t_map - p_t_map)/t_map)*100, 4),
                         }
        final_result.append(final_result_)
        header = ["now_time", "victim", "dataset", "noise_percentage", "i_map", "t_map", "fooling_rate",
                      "p_i_map", "p_t_map", "d_i_t", "d_t_i"]

        with open(final_log_save_path + '/all_final_results.csv', 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()
            writer.writerows(final_result)