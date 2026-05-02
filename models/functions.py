import torch

def mixup(inputs:torch.Tensor, targets, num_classes, alpha=10.0):

    device = inputs.device
    batch_size = inputs.size(0)
    mixed_inputs = inputs.clone()
    targets = torch.nn.functional.one_hot(targets, num_classes).to(inputs)
    mixed_targets = targets.clone()

    size = 0
    for i in range(batch_size):
        current_class = targets[i]
        different_class_idx = (targets != current_class).nonzero(as_tuple=True)[0]
        
        if len(different_class_idx) == 0:
            continue
        
        size += 1
        random_idx = different_class_idx[torch.randint(0, len(different_class_idx), (1,))]

        x1, y1 = inputs[i], targets[i]
        x2, y2 = inputs[random_idx], targets[random_idx]

        lam = torch.distributions.Beta(alpha, alpha).sample(()).to(device)

        mixed_inputs[i] = lam * x1 + (1 - lam) * x2
        mixed_targets[i] = lam * y1 + (1 - lam) * y2

    return mixed_inputs[:size+1], mixed_targets[:size+1]


def cutmix(inputs: torch.tensor, targets, num_classes, alpha=10.0):
    device = inputs.device
    batch_size = inputs.size(0)
    mixed_inputs = inputs.clone()
    targets = torch.nn.functional.one_hot(targets, num_classes).to(inputs)
    mixed_targets = targets.clone()

    size = 0
    for i in range(batch_size):
        h, w = inputs.size(2), inputs.size(3)
        lam = torch.distributions.Beta(alpha, alpha).sample(()).to(device)
        
        cut_rat = torch.sqrt(1. - lam)
        cut_w = torch.floor(w * cut_rat).to(torch.int)
        cut_h = torch.floor(h * cut_rat).to(torch.int)

        cx = torch.randint(w, (1,)).item()
        cy = torch.randint(h, (1,)).item()

        bbx1 = torch.clamp(cx - cut_w // 2, 0, w).item()
        bby1 = torch.clamp(cy - cut_h // 2, 0, h).item()
        bbx2 = torch.clamp(cx + cut_w // 2, 0, w).item()
        bby2 = torch.clamp(cy + cut_h // 2, 0, h).item()

        current_class = targets[i]
        different_class_idx = (targets != current_class).nonzero(as_tuple=True)[0]
        if len(different_class_idx) == 0:
            continue
        size += 1
        random_idx = different_class_idx[torch.randint(0, len(different_class_idx), (1,))]

        mixed_inputs[i, :, bby1:bby2, bbx1:bbx2] = inputs[random_idx, :, bby1:bby2, bbx1:bbx2]
        area = (bbx2 - bbx1) * (bby2 - bby1)
        lam = area / (h * w)
        
        mixed_targets[i] = lam * targets[i] + (1. - lam) * targets[random_idx]

    return mixed_inputs[:size+1], mixed_targets[:size+1]



