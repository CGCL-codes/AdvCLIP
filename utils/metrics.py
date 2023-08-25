import torch
import numpy as np
import torch.nn.functional as F
from torch import nn

def KL(P,Q,mask=None):
    eps = 0.0000001
    d = (P+eps).log()-(Q+eps).log()
    d = P*d
    if mask !=None:
        d = d*mask
    return torch.sum(d)
def CE(P,Q,mask=None):
    return KL(P,Q,mask)+KL(1-P,1-Q,mask)

def algorithm2(P,Q,Y):
    eps = 0.0000001
    mean = P.mean(dim=1)
    mask1 = P>=mean
    mask2 = Y == Y.t()
    mask = mask1 == mask2
    loss =torch.mean(P * torch.log((P + eps) / (Q + eps)))
    return loss

def umap(output_net, target_net, eps=0.0000001):
    # Normalize each vector by its norm
    (n, d) = output_net.shape
    output_net_norm = torch.sqrt(torch.sum(output_net ** 2, dim=1, keepdim=True))
    output_net = output_net / (output_net_norm + eps)
    output_net[output_net != output_net] = 0
    target_net_norm = torch.sqrt(torch.sum(target_net ** 2, dim=1, keepdim=True))
    target_net = target_net / (target_net_norm + eps)
    target_net[target_net != target_net] = 0
    # Calculate the cosine similarity
    model_similarity = torch.mm(output_net, output_net.transpose(0, 1))
    model_distance = 1-model_similarity #[0,2]
    model_distance[range(n), range(n)] = 3
    model_distance = model_distance - torch.min(model_distance, dim=1)[0].view(-1, 1)
    model_distance[range(n), range(n)] = 0
    model_similarity = 1-model_distance
    target_similarity = torch.mm(target_net, target_net.transpose(0, 1))
    target_distance = 1-target_similarity
    target_distance[range(n), range(n)] = 3
    target_distance = target_distance - torch.min(target_distance,dim=1)[0].view(-1,1)
    target_distance[range(n), range(n)] = 0
    target_similarity = 1 - target_distance
    # Scale cosine similarity to 0..1
    model_similarity = (model_similarity + 1.0) / 2.0
    target_similarity = (target_similarity + 1.0) / 2.0
    # Transform them into probabilities
    model_similarity = model_similarity / torch.sum(model_similarity, dim=1, keepdim=True)
    target_similarity = target_similarity / torch.sum(target_similarity, dim=1, keepdim=True)
    # Calculate the KL-divergence
    loss = CE(target_similarity,model_similarity)
    return loss


def supervised_umap(output_net, target_net,y,sample_weight=0, eps=0.0000001):
    # Normalize each vector by its norm
    (n, d) = output_net.shape
    distance = 2.0
    tahn =  nn.Tanh()
    sample_weight = sample_weight.view(-1,n)
    sample_weight_matrix = (sample_weight+sample_weight.t())/32.0
    sample_weight_matrix = tahn(sample_weight_matrix)
    y  = y.view(-1,n)
    # print("y",y)
    mask =1- (y==y.t()).float()
    mask[mask == 0] = -1
    distance = distance*mask*sample_weight_matrix
    output_net_norm = torch.sqrt(torch.sum(output_net ** 2, dim=1, keepdim=True))
    output_net = output_net / (output_net_norm + eps)
    output_net[output_net != output_net] = 0

    target_net_norm = torch.sqrt(torch.sum(target_net ** 2, dim=1, keepdim=True))
    target_net = target_net / (target_net_norm + eps)
    target_net[target_net != target_net] = 0

    # Calculate the cosine similarity
    model_similarity = torch.mm(output_net, output_net.transpose(0, 1))
    model_distance = 1 - model_similarity  # [0,2]
    model_distance[range(n), range(n)] = 100000
    model_distance = model_distance - torch.min(model_distance, dim=1)[0].view(-1, 1)
    model_distance[range(n), range(n)] = 0
    model_distance = torch.clamp(model_distance, 0+eps, 2.0-eps)

    model_similarity = 1 - model_distance

    target_similarity = torch.mm(target_net, target_net.transpose(0, 1))
    target_distance = 1 - target_similarity
    target_distance[range(n), range(n)] = 100000
    p = torch.min(target_distance, dim=1)
    target_distance = target_distance - p[0].view(-1, 1)
    target_distance[range(n), range(n)] = 0
    target_distance = (1 - sample_weight_matrix) * target_distance + distance
    target_distance = torch.clamp(target_distance, 0+eps, 2.0-eps)
    target_similarity = 1 - target_distance

    # Scale cosine similarity to 0..1
    model_similarity = (model_similarity + 1.0) / 2.0
    target_similarity = (target_similarity + 1.0) / 2.0

    # # Transform them into probabilities
    model_similarity = model_similarity / torch.sum(model_similarity, dim=1, keepdim=True)
    target_similarity = target_similarity / torch.sum(target_similarity, dim=1, keepdim=True)
    # Calculate the CE-Loss
    loss = CE(target_similarity, model_similarity)
    return loss

def cosine_similarity_loss(output_net, target_net, eps=0.0000001):
    # Normalize each vector by its norm
    output_net_norm = torch.sqrt(torch.sum(output_net ** 2, dim=1, keepdim=True))
    output_net = output_net / (output_net_norm + eps)
    output_net[output_net != output_net] = 0

    target_net_norm = torch.sqrt(torch.sum(target_net ** 2, dim=1, keepdim=True))
    target_net = target_net / (target_net_norm + eps)
    target_net[target_net != target_net] = 0

    # Calculate the cosine similarity
    model_similarity = torch.mm(output_net, output_net.transpose(0, 1))
    target_similarity = torch.mm(target_net, target_net.transpose(0, 1))

    # Scale cosine similarity to 0..1
    model_similarity = (model_similarity + 1.0) / 2.0
    target_similarity = (target_similarity + 1.0) / 2.0

    # Transform them into probabilities
    model_similarity = model_similarity / torch.sum(model_similarity, dim=1, keepdim=True)
    target_similarity = target_similarity / torch.sum(target_similarity, dim=1, keepdim=True)

    # Calculate the KL-divergence
    loss = CE(target_similarity, model_similarity)

    return loss



class dual_softmax_loss(nn.Module):
    def __init__(self, ):
        super(dual_softmax_loss, self).__init__()

    def forward(self, sim_matrix, temp=1000):
        sim_matrix = sim_matrix * F.softmax(sim_matrix / temp, dim=0) * len(
            sim_matrix)  # With an appropriate temperature parameter, the model achieves higher performance
        logpt = F.log_softmax(sim_matrix, dim=-1)  # row softmax and column softmax
        logpt = torch.diag(logpt)
        loss = -logpt
        return loss


def log_sum_exp(x):
    '''Utility function for computing log-sun-exp while determining
    This will be used to determine unaveraged confidence loss across all examples in a batch.
    '''
    x_max = x.data.max()
    return torch.log(torch.sum(torch.exp(x - x_max), -1, keepdim=True)) + x_max


def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X


def calcdist(img, txt):
    '''
    Input img = (batch,dim), txt = (batch,dim)
    Output Euclid Distance Matrix = Tensor(batch,batch), and dist[i,j] = d(img_i,txt_j)
    '''
    dist = img.unsqueeze(1) - txt.unsqueeze(0)
    dist = torch.sum(torch.pow(dist, 2), dim=2)
    return torch.sqrt(dist)


def calcmatch(label):
    '''
    Input label = (batch,)
    Output Match Matrix =Tensor(batch,batch) and match[i,j] == 1 iff. label[i]==label[j]
    '''
    match = label.unsqueeze(1) - label.unsqueeze(0)
    match[match != 0] = 1
    return 1 - match


def calcneg(dist, label, anchor, positive):
    '''
    Input dist = (batch,batch), label = (batch,), anchor = index, positive = index
    Output chosen negative sample index
    '''

    standard = dist[anchor, positive]  # positive distance
    dist = dist[anchor] - standard  # distance of other samples
    if max(dist[label != label[anchor]]) >= 0:  # there exists valid negative
        dist[dist < 0] = max(dist) + 2  # delete negative samples below standard
        dist[label == label[anchor]] = max(dist) + 2  # delete positive samples
        return int(torch.argmin(dist).cpu())  # return the closest negative sample
    else:  # choose argmax
        dist[label == label[anchor]] = min(dist) - 2  # delete positive samples
        return int(torch.argmax(dist).cpu())


def calcneg_dot(img, txt, match, anchor, positive):
    '''
    Input img = (batch,dim), txt = (batch,dim), match = (batch,batch), anchor = index, positive = index
    Output chosen negative sample index
    '''
    distdot = torch.sum(torch.mul(img.unsqueeze(1), txt.unsqueeze(0)), 2)
    distdot[match == 1] = -66666
    return int(torch.argmax(distdot[anchor]).cpu())


def Triplet(img, txt, label):
    '''
    Input img = (batch,dim), txt = (batch,dim), label = (batch,)
    Output dist = (batch,batch),match = (batch,batch), triplets = List with shape(pairs,3)
    '''
    triplet_list = []
    batch = img.shape[0]
    dist = calcdist(img, txt)
    match = calcmatch(label)
    match_n = match.cpu().numpy()
    positive_list = np.argwhere(match_n == 1).tolist()  # the index list of all positive samples
    for positive in positive_list:
        negative = calcneg(dist, label, positive[0], positive[1])  # calculate negatives
        # negative = calcneg_dot(img, txt, match, anchor, positive)  # calculate negative with dot  效果很差
        triplet_list.append([positive[0], int(positive[1].cuda()), negative])

    return dist, match, triplet_list


def Positive(img, txt, label):
    '''
    Input img = (batch,dim), txt = (batch,dim), label = (batch,)
    Output dist = (batch,batch),match = (batch,batch), positives = List with shape(pairs,2)
    Remark: return (anchor,positive) without finding triplets
    '''
    batch = img.shape[0]
    dist = calcdist(img, txt)
    match = calcmatch(label)
    sample_list = torch.tensor([x for x in range(batch)]).int().cuda()
    positive_list = [[i, int(j.cpu())] for i in range(batch) for j in sample_list[label == label[i]]]
    return dist, match, positive_list


def Modality_invariant_Loss(img, txt, label, margin=0.2):
    '''
    Input img = (batch,dim), txt = (batch,dim), label = (batch,), margin = parameter
    Calculate invariant loss between images and texts belonging to the same class
    '''
    batch = img.shape[0]
    dist = calcdist(img, txt)
    dist = torch.pow(dist, 2)
    match = calcmatch(label)  # similar is 1, dissimilar is 0
    pos = torch.mul(dist, match)
    loss = torch.sum(pos)

    return loss / batch


def Contrastive_Loss(img, txt, label, margin=0.2):
    '''
    Input img = (batch,dim), txt = (batch,dim), label = (batch,), margin = parameter
    Calculate triplet loss
    '''
    batch = img.shape[0]
    dist = calcdist(img, txt)
    dist = torch.pow(dist, 2)
    match = calcmatch(label)  # similar is 1, dissimilar is 0
    pos = torch.mul(dist, match)
    neg = margin - torch.mul(dist, 1 - match)
    neg = torch.clamp(neg, 0)
    loss = torch.sum(pos) + torch.sum(neg)

    return loss / batch


def Triplet_Loss(img, txt, label, margin=0.2, semi_hard=True):
    '''
    Input img = (batch,dim), txt = (batch,dim), label = (batch,), margin = parameter
    Calculate triplet loss
    '''
    loss = 0
    dist = calcdist(img, txt)
    dist = torch.pow(dist, 2)
    match = calcmatch(label)
    match_n = match.cpu().numpy()
    positive = np.argwhere(match_n == 1).tolist()  # the index list of all positive samples
    for x in positive:
        # # Semi-Hard Negative Mining
        if semi_hard:
            neg_index = torch.where(
                match[x[0]] == 0)  # the index list of all negative samples (shared by image and text)
            neg_dis = dist[x[0]][neg_index]
            tmp = dist[x[0], x[1]] - neg_dis + margin
            tmp = torch.clamp(tmp, 0)
            loss = loss + torch.sum(tmp, dim=-1)
        else:
            # Hard Negative Mining
            negative = calcneg(dist, label, x[0], x[1])  # calculate hard negative
            tmp = dist[x[0], x[1]] - dist[x[0], negative] + margin
            if tmp > 0:
                loss = loss + tmp

    return loss / len(positive)


def Lifted_Loss(img, txt, label, margin=1):  # the margin is set to be 1 as the original paper
    '''
    Input img = (batch,dim), txt = (batch,dim), label = (batch,), margin = parameter
    Calculate lifted structured embedding loss
    '''
    # dist, match, positive = Positive(img, txt, label)
    dist = calcdist(img, txt)
    match = calcmatch(label)
    match_n = match.cpu().numpy()
    positive = np.argwhere(match_n == 1).tolist()  # the index list of all positive samples
    loss = 0
    for x in positive:
        neg_index = torch.where(match[x[0]] == 0)  # the index list of all negative samples (shared by image and text)
        neg_dis_anchor = dist[x[0]][neg_index]
        neg_dis_postive = dist[x[1]][neg_index]
        tmp = dist[x[0], x[1]] + log_sum_exp(margin - neg_dis_postive) + log_sum_exp(margin - neg_dis_anchor)
        loss = loss + tmp

    return loss / (2 * len(positive))


def Npairs(img, txt, label, margin=0.2, alpha=0.1):
    '''
    Input img = (batch,dim), txt = (batch,dim), label = (batch,), margin = parameter, alpha = parameter
    Calculate N-pairs loss
    '''
    # dist, match, positive = Positive(img, txt, label)
    batch = img.shape[0]
    distdot_it = torch.exp(F.linear(img, txt))
    # distdot_ti = torch.t(distdot)
    # !!!!!!!!!!!!!
    distdot_ti = torch.t(distdot_it)
    # !!!!!!!!!!!!!
    match = calcmatch(label)
    match_n = match.cpu().numpy()
    positive = np.argwhere(match_n == 1).tolist()  # the index list of all positive samples
    loss = 0
    for x in positive:
        neg_index = torch.where(match[x[0]] == 0)  # the index list of all negative samples
        tmp_i2t = distdot_it[x[0], x[1]] - log_sum_exp(distdot_it[x[0]][neg_index])
        tmp_t2i = distdot_ti[x[0], x[1]] - log_sum_exp(distdot_ti[x[0]][neg_index])
        loss = loss + (tmp_i2t + tmp_t2i) / 2
    loss = -loss / len(positive)
    for x in range(batch):
        loss = loss + alpha * (torch.norm(img[x]) + torch.norm(txt[x])) / batch

    return loss


def Supervised_Contrastive_Loss(img, txt, label):
    '''
    Input img = (batch,dim), txt = (batch,dim), label = (batch,), margin = parameter
    An unofficial implementation of supervised contrastive loss for multimodal learning
    '''
    loss = 0
    batch = img.shape[0]
    dist = calcdist(img, txt)
    dist = torch.pow(dist, 2)
    dist = dist / (torch.sum(dist) / (batch * batch))  # scale the metric
    match = calcmatch(label)  # 相似为1，不相似为0
    match_n = match.cpu().numpy()
    positive = np.argwhere(match_n == 1).tolist()  # the index list of all positive samples
    loss = 0
    for x in positive:
        neg_index = torch.where(match[x[0]] == 0)  # the index list of all negative samples
        pos_sim = -dist[x[0], x[1]]
        neg_sims = -dist[x[0]][neg_index]
        tmp = pos_sim - log_sum_exp(neg_sims)

        loss = loss + tmp

    loss = -loss / len(positive)

    return loss


def regularization(features, centers, labels):
    # features = l2norm(features, dim=-1)
    distance = (features - centers[labels])
    distance = torch.sum(torch.pow(distance, 2), 1, keepdim=True)
    distance = (torch.sum(distance, 0, keepdim=True)) / features.shape[0]

    return distance


def PAN(features, centers, labels, add_regularization=False):
    """The prototype contrastive loss and regularization loss in
    PAN(https://dl.acm.org/doi/abs/10.1145/3404835.3462867)"""
    batch = features.shape[0]
    features_square = torch.sum(torch.pow(features, 2), 1, keepdim=True)  # 在第一个维度上平方
    centers_square = torch.sum(torch.pow(torch.t(centers), 2), 0, keepdim=True)
    features_into_centers = 2 * torch.matmul(features, torch.t(centers))
    dist = -(features_square + centers_square - features_into_centers)
    output = F.log_softmax(dist, dim=1)
    dce_loss = F.nll_loss(output, labels)

    if add_regularization:
        reg_loss = regularization(features, centers, labels)
        loss = dce_loss + reg_loss

    loss = dce_loss

    return loss / batch


def Label_Regression_Loss(view1_predict, view2_predict, label_onehot):
    loss = ((view1_predict - label_onehot.float()) ** 2).sum(1).sqrt().mean() + (
            (view2_predict - label_onehot.float()) ** 2).sum(1).sqrt().mean()

    return loss


def Proxy_NCA(features, label, proxies, mrg=0.1, alpha=1):
    """
    Input:
    :param feature: [2*batch, dim]  concat image and text features
    :param label: [2*batch]
    :param proxies: [feature_dim, n_classes]
    :return: Proxy Anchor loss
 """
    P = torch.t(proxies)  # [feature_dim, n_classes]-->[n_classes, feature_dim]
    n_classes = P.shape[0]
    # similar to Proxc-NCA and Normlized Softmax
    cos = F.linear(features, P)  # Calcluate cosine similarity [batch, n_classes]

    # Proxy-NCA loss (similar to Normlized Softmax and PAN，while the denominator does not contain positive prototype)
    loss = 0
    for x in range(features.shape[0]):
        pos = torch.exp(cos[x, label[x]])
        neg = torch.exp(cos[x]).sum(dim=-1) - pos
        loss = loss + torch.log(pos / neg)
    loss = -loss / features.shape[0]

    return loss


def Proxy_Anchor(features, label, proxies, mrg=0.1, alpha=1):
    """
    Input:
    :param
    feature: [2 * batch, dim]
    concat
    image and text
    features
    :param
    label: [2 * batch]
    :param
    proxies: [feature_dim, n_classes]
    :return: Proxy
    Anchor
    loss
    """
    P = torch.t(proxies)  # [feature_dim, n_classes]-->[n_classes, feature_dim] 
    n_classes = P.shape[0]
    # similar to Proxc-NCA and Normlized Softmax
    cos = F.linear(features, P)  # Calcluate cosine similarity [batch, n_classes]

    P_one_hot = label  # [batch, n_classes]
    N_one_hot = 1 - P_one_hot

    pos_exp = torch.exp(-alpha * (cos - mrg))  # [batch, n_class]
    neg_exp = torch.exp(alpha * (cos + mrg))  # 出现了e+30导致nan

    with_pos_proxies = torch.nonzero(P_one_hot.sum(dim=0) != 0).squeeze(
        dim=1)  # The set of positive proxies of data in the batch
    num_valid_proxies = len(with_pos_proxies)  # The number of positive proxies

    P_sim_sum = torch.where(P_one_hot == 1, pos_exp, torch.zeros_like(pos_exp)).sum(dim=0)
    N_sim_sum = torch.where(N_one_hot == 1, neg_exp, torch.zeros_like(neg_exp)).sum(dim=0)

    pos_term = torch.log(1 + P_sim_sum).sum() / num_valid_proxies
    neg_term = torch.log(1 + N_sim_sum).sum() / n_classes
    loss = pos_term + neg_term

    return loss
