from config import ALPHA, EPSILON, MAX_ITERS, clamp_min, clamp_max, device

import torch
import torch.nn.functional as F


@torch.no_grad()
def eval_clean(model, loader):
    model.eval()
    correct = total = 0
    for original, view1, view2, ids, labels in loader:
        x, y = original.to(device), labels.to(device)
        _, logits = model(x)  # Access the logits from the tuple
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return correct / total

def fgsm(model, X, y, epsilon=8/255):
    """ Construct FGSM adversarial examples on the examples X"""
    model.zero_grad()
    delta = torch.zeros_like(X, requires_grad=True)
    logits = model(X + delta)
    loss = F.cross_entropy(logits, y)
    loss.backward()
    return epsilon * delta.grad.detach().sign()

def eval_fgsm(model, loader, epsilon=8/255):
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        # craft one‐step attack (do NOT use no_grad here)
        delta = fgsm(model, x, y, epsilon)
        x_adv = (x + delta).clamp(clamp_min, clamp_max)
        with torch.no_grad():
            logits = model(x_adv)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total

def pgd_attack(model, images, labels, epsilon=EPSILON, alpha=ALPHA, iters=MAX_ITERS):
    images = images.clone().detach().to(device)
    images.requires_grad = True # Ensure gradients are tracked from the beginning
    labels = labels.to(device)
    ori_images = images.clone().detach()

    model.eval()
    for _ in range(iters):
        # images.requires_grad = True # Remove this line
        _, outputs = model(images)
        loss = F.cross_entropy(outputs, labels)
        grads = torch.autograd.grad(loss, images, retain_graph=False, create_graph=False)[0]
        adv_images = images + alpha * grads.sign()
        eta = torch.clamp(adv_images - ori_images, min=-epsilon, max=epsilon)
        images = torch.clamp(ori_images + eta, 0, 1) # Remove .detach() here

    return images.detach() # Detach only the final adversarial image

def eval_pgd(model, loader, epsilon=EPSILON, alpha=ALPHA, iters=MAX_ITERS):
    model.eval()
    correct = total = 0
    # Remove torch.no_grad() here as gradients are needed for PGD attack
    for x, _, _, _, y in loader:
        x, y = x.to(device), y.to(device)
        x_adv = pgd_attack(model, x, y, epsilon, alpha, iters)
        _, logits = model(x_adv)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return correct / total