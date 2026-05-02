import torch
import torch.nn as nn
from torch.nn import functional as F

class AngularPenaltySMLoss(nn.Module):
    def __init__(self, loss_type='cosface', eps=1e-7, s=20, m=0):
        super(AngularPenaltySMLoss, self).__init__()
        loss_type = loss_type.lower()
        assert loss_type in ['arcface', 'sphereface', 'cosface', 'crossentropy']
        if loss_type == 'arcface':
            self.s = 64.0 if not s else s
            self.m = 0.5 if not m else m
        if loss_type == 'sphereface':
            self.s = 64.0 if not s else s
            self.m = 1.35 if not m else m
        if loss_type == 'cosface':
            self.s = 20.0 if not s else s
            self.m = 0.0 if not m else m
        self.loss_type = loss_type
        self.eps = eps

        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, wf, labels, return_type='batch_mean'):
        if self.loss_type == 'crossentropy':
            return self.cross_entropy(wf, labels)
        else:
            if self.loss_type == 'cosface':
                numerator = self.s * (torch.diagonal(wf.transpose(0, 1)[labels]) - self.m)
            if self.loss_type == 'arcface':
                numerator = self.s * torch.cos(torch.acos(
                    torch.clamp(torch.diagonal(wf.transpose(0, 1)[labels]), -1. + self.eps, 1 - self.eps)) + self.m)
            if self.loss_type == 'sphereface':
                numerator = self.s * torch.cos(self.m * torch.acos(
                    torch.clamp(torch.diagonal(wf.transpose(0, 1)[labels]), -1. + self.eps, 1 - self.eps)))

            excl = torch.cat([torch.cat((wf[i, :y], wf[i, y + 1:])).unsqueeze(0) for i, y in enumerate(labels)], dim=0)
            denominator = torch.exp(numerator) + torch.sum(torch.exp(self.s * excl), dim=1)
            L = numerator - torch.log(denominator)

            if return_type == 'batch_mean':
                return -torch.mean(L)
            elif return_type == 'per_sample':
                return -L
        

class MahalanobisLoss(nn.Module):
    def __init__(self, sigma_old):

        super(MahalanobisLoss, self).__init__()
        self.sigma_inv = sigma_old

    def compute_mahalanobis_distance(self, embedding1, embedding2, cov_inv):

        embedding1 = F.normalize(embedding1, p=2, dim=0)
        embedding2 = F.normalize(embedding2, p=2, dim=0)

        delta = embedding1 - embedding2
        cov_inv = cov_inv + 1e-6 * torch.eye(cov_inv.size(0), device=cov_inv.device)
        intermediate = torch.matmul(cov_inv, delta.unsqueeze(1))
        distance = torch.matmul(delta.unsqueeze(0), intermediate)
        distance = torch.sqrt(distance + 1e-8).squeeze()
        return distance

    def forward(self, x, y, labels):
        unique_labels = torch.unique(labels)
        loss = 0.0

        for label in unique_labels:

            class_mask = (labels == label)
            x_class = x[class_mask]  
            y_class = y[class_mask]  
            cov_inv = self.sigma_inv[label.item()]  

            for i in range(x_class.size(0)):
                for j in range(i + 1, x_class.size(0)):
                    d_old = self.compute_mahalanobis_distance(x_class[i], x_class[j], cov_inv)
                    d_new = self.compute_mahalanobis_distance(y_class[i], y_class[j], cov_inv)
                    loss += (d_old - d_new).abs()

        return loss / x.size(0) 
    
def compute_angle_weighted_patch_distillation_loss(p_n, p_o, cls_n):

    p_n_normalized = F.normalize(p_n, p=2, dim=-1)  # [batch_size, num_tokens, embed_dim]
    p_o_normalized = F.normalize(p_o, p=2, dim=-1)  # [batch_size, num_tokens, embed_dim]

    alpha_cos = F.cosine_similarity(cls_n.unsqueeze(1),p_n,dim=-1) # [batch_size, num_tokens]
    alpha_cos = alpha_cos.clamp(min=-1.0, max=1.0)  
    alpha_angle = 1 - (torch.acos(alpha_cos) / torch.pi)  # [batch_size, num_tokens]

    distances = torch.norm(p_n_normalized - p_o_normalized, p=2, dim=-1)  # [batch_size, num_tokens]
    weighted_distances = (1-alpha_angle.detach()) * distances  # [batch_size, num_tokens]
    loss = weighted_distances.mean()

    return loss