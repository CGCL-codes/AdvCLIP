import sys
import os
import datetime
import numpy as np
import scipy.spatial
import torch
import torch.nn.functional as F
from tqdm import tqdm
from collections import Counter


def make_print_to_file(path='./'):
    '''
    path， it is a path for save your log about fuction print
    example:
    use  make_print_to_file()   and the   all the information of funtion print , will be write in to a log file
    :return:
    '''
    class Logger(object):
        def __init__(self, filename="Default.log", path="./"):
            self.terminal = sys.stdout
            self.log = open(os.path.join(path, filename), "w", encoding='utf8', )

        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)

        def flush(self):
            pass

    fileName = datetime.datetime.now().strftime('%Y_%m_%d')
    sys.stdout = Logger(fileName + '.log', path=path)
    # print(fileName.center(60, '*'))
    return fileName


def fx_calc_map_label(image, text, label, k=1, dist_method='COS'):
    if dist_method == 'L2':
        dist = scipy.spatial.distance.cdist(image, text, 'euclidean')
    elif dist_method == 'COS':
        dist = scipy.spatial.distance.cdist(image, text, 'cosine')
    ord = dist.argsort() # [batch, batch]
    numcases = dist.shape[0]
    if k == 0:
      k = numcases
    res = []
    for i in range(numcases):
        order = ord[i]
        p = 0.0
        r = 0.0
        for j in range(k):
            if label[i] == label[order[j]]:
                r += 1
                p += (r / (j + 1))
        if r > 0:
            res += [p / r]
        else:
            res += [0]
    map = round(np.mean(res), 4)
    return map


def fx_calc_recall(image, text, label, k=0, dist_method='L2'):
    if dist_method == 'L2':
        dist = scipy.spatial.distance.cdist(image, text, 'euclidean')
    elif dist_method == 'COS':
        dist = scipy.spatial.distance.cdist(image, text, 'cosine')

    ord = dist.argsort() # [batch, batch]
    ranks = np.zeros(image.shape[0])

    # R@K
    for i in range(image.shape[0]):
        q_label = label[i]
        r_labels = label[ord[i]]
        ranks[i] = np.where(r_labels == q_label)[0][0]
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)

    return r1, r5, r10

def calc_mean(label, res, num_class):
    num_list = [0 for i in range(10)]
    value_list = [0 for i in range(10)]
    for i in range(len(res)):
        num_list[label[i]] += 1
        value_list[label[i]] += res[i]
    for i in range(num_class):
        if num_list[i] != 0:
            value_list[i] = value_list[i]/num_list[i]
            value_list[i] = round(value_list[i], 4)
        else:
            value_list[i] = 0
    return value_list

@torch.no_grad()
def extract_features(loader, model):
    model.eval()
    backbone_features, labels = [], []
    for im, text, lab, id in tqdm(loader):
        im = im.cuda(non_blocking=True)
        lab = lab.cuda(non_blocking=True)
        outs = model.encode_image(im.squeeze())
        backbone_features.append(outs)
        labels.append(lab)
    model.train()
    backbone_features = torch.cat(backbone_features)
    labels = torch.cat(labels)
    return backbone_features,  labels

@torch.no_grad()
def extract_patch_features(loader, model, p, mask):
    model.eval()
    backbone_features, labels = [], []
    for im, text, lab, id in tqdm(loader):
        new_shape = im.shape
        im = torch.mul(mask.type(torch.FloatTensor), p.type(torch.FloatTensor)) + torch.mul(
            1 - mask.expand(new_shape).type(torch.FloatTensor), im.type(torch.FloatTensor))

        im = im.cuda(non_blocking=True)
        lab = lab.cuda(non_blocking=True)
        outs = model.encode_image(im.squeeze())
        backbone_features.append(outs)
        labels.append(lab)
    model.train()
    backbone_features = torch.cat(backbone_features)
    labels = torch.cat(labels)
    return backbone_features,  labels

@torch.no_grad()
def extract_per_features(loader, model, p):
    model.eval()
    backbone_features, labels = [], []
    for im, text, lab, id in tqdm(loader):
        new_shape = im.shape
        im = im.cuda(non_blocking=True)
        lab = lab.cuda(non_blocking=True)
        im = im + p.expand(new_shape)
        outs = model.encode_image(im.squeeze())
        backbone_features.append(outs)
        labels.append(lab)
    model.train()
    backbone_features = torch.cat(backbone_features)
    labels = torch.cat(labels)
    return backbone_features,  labels

def evaluate(model, loader, device):
    model.eval()
    t_imgs, t_txts, t_labels = [], [], []
    with torch.no_grad():
        for img, text, labels, id in loader:
            img = img.squeeze().to(device)
            text = text.squeeze().to(device)
            with torch.no_grad():
                image_features = model.encode_image(img)
                text_features = model.encode_text(text)
                t_imgs.append(image_features.cpu().numpy())
                t_txts.append(text_features.cpu().numpy())
                t_labels.append(labels.cpu().numpy())

    t_imgs = np.concatenate(t_imgs)  # for visualization
    t_txts = np.concatenate(t_txts)  # for visualization
    t_labels = np.concatenate(t_labels)
    i_map = fx_calc_map_label(t_imgs, t_txts, t_labels)
    t_map = fx_calc_map_label(t_txts, t_imgs, t_labels)

    return i_map, t_map, t_imgs, t_txts, t_labels

def adv_evaluate(model, loader, uap_noise, mask, device):
    model.eval()
    t_imgs, t_txts, t_labels = [], [], []
    with torch.no_grad():
        for img, text, labels, id in loader:
            img = img.squeeze().to(device)
            # img = img + uap_noise.expand(img.shape)
            img = torch.mul(mask.type(torch.FloatTensor),
                             uap_noise.type(torch.FloatTensor)) + torch.mul(
                1 - mask.expand(img.shape).type(torch.FloatTensor), img.type(torch.FloatTensor))

            text = text.squeeze().to(device)
            with torch.no_grad():
                image_features = model.encode_image(img.to(device))
                text_features = model.encode_text(text)
                t_imgs.append(image_features.cpu().numpy())
                t_txts.append(text_features.cpu().numpy())
                t_labels.append(labels.cpu().numpy())

    t_imgs = np.concatenate(t_imgs)  # for visualization
    t_txts = np.concatenate(t_txts)  # for visualization
    t_labels = np.concatenate(t_labels)
    i_map = fx_calc_map_label(t_imgs, t_txts, t_labels)
    t_map = fx_calc_map_label(t_txts, t_imgs, t_labels)

    return i_map, t_map, t_imgs, t_txts, t_labels

def knn_patch_fr(train_loader, test_loader, model, p, mask, k=20, T=0.07):
    # extract train features
    train_features, train_targets = extract_features(train_loader, model)

    # extract test features
    test_features, test_targets = extract_features(test_loader, model)

    # extract per_test features
    p_test_features, p_test_targets = extract_patch_features(test_loader, model, p, mask)


    max_distance_matrix_size = int(5e6)

    train_features = F.normalize(train_features)
    test_features = F.normalize(test_features)
    p_test_features = F.normalize(p_test_features)


    num_classes = torch.unique(test_targets).numel()

    num_test_images = test_targets.size(0)
    num_p_test_images = p_test_targets.size(0)
    num_train_images = train_targets.size(0)

    chunk_size = min(
        max(1, max_distance_matrix_size // num_train_images),
        num_test_images,
    )

    k = min(k, num_train_images)

    # test clean
    top1, top5, total = 0.0, 0.0, 0
    retrieval_one_hot = torch.zeros(k, num_classes).to(train_features.device)
    p_top1, p_top5, p_total = 0.0, 0.0, 0
    p_retrieval_one_hot = torch.zeros(k, num_classes).to(train_features.device)
    fr = 0.0

    for idx in range(0, num_test_images, chunk_size):
        # get the features for test images
        features = test_features[idx: min((idx + chunk_size), num_test_images), :]

        targets = test_targets[idx: min((idx + chunk_size), num_test_images)]
        batch_size = targets.size(0)

        # calculate the dot product and compute top-k neighbors
        similarities = torch.mm(features, train_features.t())

        similarities, indices = similarities.topk(k, largest=True, sorted=True)
        candidates = train_targets.view(1, -1).expand(batch_size, -1)
        retrieved_neighbors = torch.gather(candidates, 1, indices)

        retrieval_one_hot.resize_(batch_size * k, num_classes).zero_()
        retrieval_one_hot.scatter_(1, retrieved_neighbors.view(-1, 1), 1)

        similarities = similarities.clone().div_(T).exp_()

        probs = torch.sum(
            torch.mul(
                retrieval_one_hot.view(batch_size, -1, num_classes),
                similarities.view(batch_size, -1, 1),
            ),
            1,
        )
        _, predictions = probs.sort(1, True)

        # get the features for test images
        p_features = p_test_features[idx: min((idx + chunk_size), num_p_test_images), :]
        p_targets = p_test_targets[idx: min((idx + chunk_size), num_p_test_images)]
        p_batch_size = p_targets.size(0)

        # calculate the dot product and compute top-k neighbors
        p_similarities = torch.mm(p_features, train_features.t())

        p_similarities, p_indices = p_similarities.topk(k, largest=True, sorted=True)
        p_candidates = train_targets.view(1, -1).expand(p_batch_size, -1)
        p_retrieved_neighbors = torch.gather(p_candidates, 1, p_indices)

        p_retrieval_one_hot.resize_(p_batch_size * k, num_classes).zero_()
        p_retrieval_one_hot.scatter_(1, p_retrieved_neighbors.view(-1, 1), 1)
        p_similarities = p_similarities.clone().div_(T).exp_()

        p_probs = torch.sum(
            torch.mul(
                p_retrieval_one_hot.view(p_batch_size, -1, num_classes),
                p_similarities.view(p_batch_size, -1, 1),
            ),
            1,
        )
        _, p_predictions = p_probs.sort(1, True)

        correct = predictions.eq(targets.data.view(-1, 1))

        top1 = top1 + correct.narrow(1, 0, 1).sum().item()
        top5 = (top5 + correct.narrow(1, 0,
                                      min(5, k, correct.size(-1))).sum().item())  # top5 does not make sense if k < 5
        total += targets.size(0)

        # find the predictions that match the target
        p_correct = p_predictions.eq(p_targets.data.view(-1, 1))
        p_top1 = p_top1 + p_correct.narrow(1, 0, 1).sum().item()
        p_top5 = (p_top5 + p_correct.narrow(1, 0, min(5, k, p_correct.size(
            -1))).sum().item())  # top5 does not make sense if k < 5

        p_total += p_targets.size(0)

        fr += predictions.eq(p_predictions).narrow(1, 0, 1).sum().item()

    top1 = top1 * 100.0 / total
    top5 = top5 * 100.0 / total

    p_top1 = p_top1 * 100.0 / p_total
    p_top5 = p_top5 * 100.0 / p_total

    fooling_rate = (total - fr) * 100.0 / float(total)
    print(top1, top5, p_top1, p_top5, fooling_rate)
    return top1, top5, p_top1, p_top5, fooling_rate

def adv_eval(model, train_loader, test_loader, uap_noise, mask, device):
    model.eval()
    model.to(device)
    uap_noise.to(device)
    total_correct, total_p_correct, total_fr, step = 0., 0., 0., 0.
    with torch.no_grad():
        # data, text, target, id
        for i, (data, text, target, id) in enumerate(tqdm(test_loader)):
            data, text, target = data.squeeze().to(device), text.squeeze().to(device), target.to(device)
            #     text = text.squeeze().to(device)
            # p_data = data + uap_noise.expand(data.shape)
            p_data = torch.mul(mask.type(torch.FloatTensor),
                             uap_noise.type(torch.FloatTensor)) + torch.mul(
                1 - mask.expand(data.shape).type(torch.FloatTensor), data.type(torch.FloatTensor))

            text_features = model.encode_text(text).float()
            text_features /= text_features.norm(dim=-1, keepdim=True)

            img_features = model.encode_image(data).float()
            img_features /= img_features.norm(dim=-1, keepdim=True)

            p_img_features = model.encode_image(p_data.to(device)).float()
            p_img_features /= img_features.norm(dim=-1, keepdim=True)


            similarity = 100. * (img_features @ text_features.T)
            p_similarity = 100. * (p_img_features @ text_features.T)

            probs = F.softmax(similarity, dim=-1).max(-1)[1]
            p_probs = F.softmax(p_similarity, dim=-1).max(-1)[1]

            total_correct += probs.eq(target).sum().item()

            total_p_correct += p_probs.eq(target).sum().item()

            total_fr += p_probs.eq(probs).sum().item()

            step += target.size(0)

        acc = total_correct / step * 100.
        p_acc = total_p_correct / step * 100.
        fr =  (step - total_fr)  / step * 100.
        return acc, p_acc, fr


##########################  CROSS TEST  ##########################


def cross_evaluate_finetune(clip_model, model, loader, device):
    model.eval()
    # clip_model, preprocess = clip.load("RN50", device=device)  # ['RN50', 'RN101', 'RN50x4', 'RN50x16', 'ViT-B/32', 'ViT-B/16']
    t_imgs, t_txts, t_labels = [], [], []
    with torch.no_grad():
        for img, text, labels, id in loader:
            text = text.to(device)
            img = img.to(device)
            text = text.to(device)

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


def cross_evaluate(clip_model, model, loader, device):
    model.eval()
    t_imgs, t_txts, t_labels = [], [], []
    with torch.no_grad():
        for img, text, labels, id in loader:
            text = text.to(device)
            img = img.to(device)
            text = text.to(device)

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

def p_cross_evaluate(clip_model, model, loader, uap_noise, device):
    model.eval()
    uap_noise.to(device)
    fr, total = 0, 0
    t_imgs, t_txts, t_labels, nor_pre, p_pre = [], [], [], [], []
    with torch.no_grad():
        for counter, (img, text, labels, id) in enumerate(tqdm(loader)):
            p_img = img + uap_noise.expand(img.shape)
            text = text.to(device)
            img = img.to(device)
            p_img = p_img.to(device)

            p_img = clip_model.encode_image(p_img.squeeze())
            text = clip_model.encode_text(text.squeeze())

            labels = labels.int().to(device)
            _, _, _, nor_img_predict, _ = model(img, text)

            _, img_feature, text_feature, img_predict, text_predict = model(p_img, text)

            nor_pre.append(nor_img_predict.cpu().numpy())
            t_imgs.append(img_feature.cpu().numpy())
            t_txts.append(text_feature.cpu().numpy())
            t_labels.append(labels.cpu().numpy())
            fr += fooling_rate(nor_img_predict, img_predict, labels)

    afr = fr / (counter + 1)
    t_imgs = np.concatenate(t_imgs)  # for visualization
    t_txts = np.concatenate(t_txts)  # for visualization
    t_labels = np.concatenate(t_labels)
    i_map = fx_calc_map_label(t_imgs, t_txts, t_labels)
    t_map = fx_calc_map_label(t_txts, t_imgs, t_labels)
    print('Image to Text: MAP: {:.4f}'.format(i_map))
    print('Text to Image: MAP: {:.4f}'.format(t_map))

    return i_map, t_map, t_imgs, t_txts, t_labels, afr


def pat_cross_evaluate(clip_model, model, loader, uap_noise, mask, device):
    model.eval()
    uap_noise.to(device)
    fr, total = 0, 0
    t_imgs, t_txts, t_labels, nor_pre, p_pre = [], [], [], [], []
    with torch.no_grad():
        for counter, (img, text, labels, id) in enumerate(tqdm(loader)):

            p_img = torch.mul(mask.type(torch.FloatTensor),
                             uap_noise.type(torch.FloatTensor)) + torch.mul(
                1 - mask.expand(img.shape).type(torch.FloatTensor), img.type(torch.FloatTensor))


            text = text.to(device)
            img = img.to(device)
            p_img = p_img.to(device)


            img = clip_model.encode_image(img.squeeze())
            p_img = clip_model.encode_image(p_img.squeeze())
            text = clip_model.encode_text(text.squeeze())
            labels = labels.int().to(device)
            _, _, _, nor_img_predict, _ = model(img, text)
            _, img_feature, text_feature, img_predict, text_predict = model(p_img, text)

            nor_pre.append(nor_img_predict.cpu().numpy())
            p_pre.append(img_predict.cpu().numpy())
            t_imgs.append(img_feature.cpu().numpy())
            t_txts.append(text_feature.cpu().numpy())
            t_labels.append(labels.cpu().numpy())
            fr += fooling_rate(nor_img_predict, img_predict, labels)

    afr = fr / (counter + 1)
    t_imgs = np.concatenate(t_imgs)  # for visualization
    t_txts = np.concatenate(t_txts)  # for visualization
    t_labels = np.concatenate(t_labels)
    i_map = fx_calc_map_label(t_imgs, t_txts, t_labels)
    t_map = fx_calc_map_label(t_txts, t_imgs, t_labels)
    print('Image to Text: MAP: {:.4f}'.format(i_map))
    print('Text to Image: MAP: {:.4f}'.format(t_map))

    return i_map, t_map, t_imgs, t_txts, t_labels, afr



def pat_cross_evaluate_finetune(clip_model, model, loader, uap_noise, mask, device):
    model.eval()
    uap_noise.to(device)
    fr, total = 0, 0
    t_imgs, t_txts, t_labels, nor_pre, p_pre = [], [], [], [], []
    with torch.no_grad():
        for counter, (img, text, labels, id) in enumerate(tqdm(loader)):

            p_img = torch.mul(mask.type(torch.FloatTensor),
                             uap_noise.type(torch.FloatTensor)) + torch.mul(
                1 - mask.expand(img.shape).type(torch.FloatTensor), img.type(torch.FloatTensor))

            text = text.to(device)
            img = img.to(device)
            p_img = p_img.to(device)

            img = clip_model.encode_image(img.squeeze())
            p_img = clip_model.encode_image(p_img.squeeze())
            text = clip_model.encode_text(text.squeeze())
            labels = labels.int().to(device)
            _, _, _, nor_img_predict, _ = model(img, text)
            _, img_feature, text_feature, img_predict, text_predict = model(p_img, text)

            nor_pre.append(nor_img_predict.cpu().numpy())
            p_pre.append(img_predict.cpu().numpy())
            t_imgs.append(img_feature.cpu().numpy())
            t_txts.append(text_feature.cpu().numpy())
            t_labels.append(labels.cpu().numpy())
            fr += fooling_rate(nor_img_predict, img_predict, labels)

    # fr = fooling_rate(nor_pre, p_pre, t_labels)
    afr = fr / (counter + 1)
    t_imgs = np.concatenate(t_imgs)  # for visualization
    t_txts = np.concatenate(t_txts)  # for visualization
    t_labels = np.concatenate(t_labels)
    i_map = fx_calc_map_label(t_imgs, t_txts, t_labels)
    t_map = fx_calc_map_label(t_txts, t_imgs, t_labels)
    print('Image to Text: MAP: {:.4f}'.format(i_map))
    print('Text to Image: MAP: {:.4f}'.format(t_map))

    return i_map, t_map, t_imgs, t_txts, t_labels, afr

def per_cross_evaluate(clip_model, model, loader, uap_noise, device):
    model.eval()
    uap_noise.to(device)
    fr, total = 0, 0
    t_imgs, t_txts, t_labels, nor_pre, p_pre = [], [], [], [], []
    with torch.no_grad():
        for counter, (img, text, labels, id) in enumerate(tqdm(loader)):
            # for img, text, labels, id in loader:
            img = img.to(device)
            p_img = img.to(device) + uap_noise.to(device).expand(img.shape)

            text = text.to(device)
            p_img = p_img.to(device)
            img = clip_model.encode_image(img.squeeze())
            p_img = clip_model.encode_image(p_img.squeeze())
            text = clip_model.encode_text(text.squeeze())
            labels = labels.int().to(device)
            _, _, _, nor_img_predict, _ = model(img, text)
            _, img_feature, text_feature, img_predict, text_predict = model(p_img, text)

            nor_pre.append(nor_img_predict.cpu().numpy())
            p_pre.append(img_predict.cpu().numpy())
            t_imgs.append(img_feature.cpu().numpy())
            t_txts.append(text_feature.cpu().numpy())
            t_labels.append(labels.cpu().numpy())
            fr += fooling_rate(nor_img_predict, img_predict, labels)

    # fr = fooling_rate(nor_pre, p_pre, t_labels)
    afr = fr / (counter + 1)
    t_imgs = np.concatenate(t_imgs)  # for visualization
    t_txts = np.concatenate(t_txts)  # for visualization
    t_labels = np.concatenate(t_labels)
    i_map = fx_calc_map_label(t_imgs, t_txts, t_labels)
    t_map = fx_calc_map_label(t_txts, t_imgs, t_labels)
    print('Image to Text: MAP: {:.4f}'.format(i_map))
    print('Text to Image: MAP: {:.4f}'.format(t_map))

    return i_map, t_map, t_imgs, t_txts, t_labels, afr



##########################  SOLO TEST  ##########################

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

def solo_test(encoder, classifier, test_loader, device):
    top1_accuracy = 0
    top5_accuracy = 0

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


def solo_adv_test(encoder, classifier, test_loader, uap, mask, device):
    top1_accuracy = 0
    top5_accuracy = 0

    classifier.eval()

    with torch.no_grad():
        for counter, (x_batch, text_batch, y_batch, id) in enumerate(tqdm(test_loader)):
            new_shape = x_batch.shape
            # choose attack type
            x_batch = torch.mul(mask.type(torch.FloatTensor), uap.type(torch.FloatTensor)) + torch.mul(1 - mask.expand(new_shape).type(torch.FloatTensor), x_batch.type(torch.FloatTensor))
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            h = encoder.encode_image(x_batch.squeeze())
            x_in = h.view(h.size(0), -1)
            # x_in = torch.tensor(x_in, dtype=torch.float)
            logits = classifier(x_in.float())
            # print(counter)
            top1, top5 = accuracy(logits, y_batch, topk=(1, 5))
            top1_accuracy += top1[0]
            top5_accuracy += top5[0]

        top1_accuracy /= (counter + 1)
        top5_accuracy /= (counter + 1)

    return top1_accuracy.item(), top5_accuracy.item()


def solo_adv_test_per(encoder, classifier, test_loader, uap, device):
    top1_accuracy = 0
    top5_accuracy = 0

    classifier.eval()

    with torch.no_grad():
        for counter, (x_batch, text_batch, y_batch, id) in enumerate(tqdm(test_loader)):
            new_shape = x_batch.shape
            # choose attack type
            x_batch = x_batch.to(device) + uap.to(device).expand(new_shape)
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            h = encoder.encode_image(x_batch.squeeze())
            x_in = h.view(h.size(0), -1)
            # x_in = torch.tensor(x_in, dtype=torch.float)
            logits = classifier(x_in.float())
            # print(counter)
            top1, top5 = accuracy(logits, y_batch, topk=(1, 5))
            top1_accuracy += top1[0]
            top5_accuracy += top5[0]

        top1_accuracy /= (counter + 1)
        top5_accuracy /= (counter + 1)

    return top1_accuracy.item(), top5_accuracy.item()



def fooling_rate(clean_output, per_output, target):
    with torch.no_grad():
        # maxk = max(topk)
        batch_size = target.size(0)

        _, c_pred = clean_output.max(1)
        c_pred = c_pred.t()

        _, p_pred = per_output.max(1)
        p_pred = p_pred.t()


        fooling_rate = float(torch.sum(c_pred != p_pred)) / float(batch_size)


        return fooling_rate * 100


def solo_fr_test(encoder, classifier, test_loader, uap, mask, device):
    fr = 0
    classifier.eval()

    with torch.no_grad():
        for counter, (x_batch, text_batch, y_batch, id) in enumerate(tqdm(test_loader)):
            new_shape = x_batch.shape

            clean_x_batch = x_batch.clone()
            clean_x_batch = clean_x_batch.to(device)
            y_batch = y_batch.to(device)

            per_x_batch = torch.mul(mask.type(torch.FloatTensor), uap.type(torch.FloatTensor)) + torch.mul(
                1 - mask.expand(new_shape).type(torch.FloatTensor), x_batch.type(torch.FloatTensor))
            per_x_batch = per_x_batch.to(device)

            c_h = encoder.encode_image(clean_x_batch.squeeze())
            p_h = encoder.encode_image(per_x_batch.squeeze())

            c_x_in = c_h.view(c_h.size(0), -1)
            p_x_in = p_h.view(p_h.size(0), -1)

            c_logits = classifier(c_x_in.float())
            p_logits = classifier(p_x_in.float())

            fooling_ra = fooling_rate(c_logits, p_logits, y_batch)
            fr += fooling_ra

        fr /= (counter + 1)

    return fr

def solo_fr_test_per(encoder, classifier, test_loader, uap, device):
    fr = 0
    classifier.eval()

    with torch.no_grad():
        for counter, (x_batch, text_batch, y_batch, id) in enumerate(tqdm(test_loader)):
            new_shape = x_batch.shape

            clean_x_batch = x_batch.clone()
            clean_x_batch = clean_x_batch.to(device)
            y_batch = y_batch.to(device)
            per_x_batch = x_batch.to(device) + uap.to(device).expand(new_shape)
            per_x_batch = per_x_batch.to(device)

            c_h = encoder.encode_image(clean_x_batch.squeeze())
            p_h = encoder.encode_image(per_x_batch.squeeze())

            c_x_in = c_h.view(c_h.size(0), -1)
            p_x_in = p_h.view(p_h.size(0), -1)

            c_logits = classifier(c_x_in.float())
            p_logits = classifier(p_x_in.float())

            fooling_ra = fooling_rate(c_logits, p_logits, y_batch)
            fr += fooling_ra

        fr /= (counter + 1)

    return fr