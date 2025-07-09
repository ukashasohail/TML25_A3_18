from config import EPSILON, ALPHA, MAX_ITERS, WEIGHT, TEMPERATURE, clamp_min, clamp_max
from loss import pairwise_similarity, NT_xent

import torch

# [4.0]
def project(x_adv, x_orig, epsilon):
    """Projects the preturbed image back into the l_infinity norm ball if it exceeds threshold measured by epsilon"""
    return torch.max(torch.min(x_adv, x_orig + epsilon), x_orig - epsilon)

class RepresentationAdversarial():
    """
    
    For the complete RoCL loss of LRoCL,θ,π + λLcon,θ,π(t(x)adv, {t'(x)}, {t(x)neg})
    This computes only the regularized loss part of λLcon,θ,π(t(x)adv, {t'(x)}, {t(x)neg})
    
    generate the adversarial example of x using a stochastically transformed image t(x), rather
    than the original image x, which will allow us to generate diverse attack samples. (Kim et al.)
    
    """
    def __init__(self, model, epsilon=EPSILON, alpha=ALPHA, min_val=clamp_min, max_val=clamp_max, max_iters=MAX_ITERS, weight=WEIGHT):
        self.model = model
        self.epsilon = epsilon
        self.alpha = alpha
        self.min_value = min_val
        self.max_value = max_val
        self.max_iters = max_iters
        self.weight = weight

    def get_loss(self, original_image, target_image, optimizer):
        """Random preturbation to image using PGD"""
        x = original_image.clone() + torch.empty_like(original_image).uniform_(-self.epsilon, self.epsilon)
        x = torch.clamp(x, self.min_value, self.max_value)
        x.requires_grad = True

        self.model.eval()
        with torch.enable_grad():
            for _ in range(self.max_iters):
                self.model.zero_grad()
                inputs = torch.cat((x, target_image))
                z, _ = self.model(inputs)
                sim, _ = pairwise_similarity(z, TEMPERATURE)
                loss = NT_xent(sim)
                grads = torch.autograd.grad(loss, x)[0]        # sgn(∇𝐿 𝑥,𝑦,𝜃 ) 
                x.data += self.alpha * torch.sign(grads.data)  # 𝑥 = 𝑥 + 𝜀 sgn(∇𝐿 𝑥,𝑦,𝜃 )
                x = torch.clamp(x, self.min_value, self.max_value)
                x = project(x, original_image, self.epsilon)   # # project back to epsilon-ball and valid pixel range

            self.model.train()
            optimizer.zero_grad()
            inputs = torch.cat((x, target_image))
            z, _ = self.model(inputs)
            sim, _ = pairwise_similarity(z, TEMPERATURE)
            loss = self.weight * NT_xent(sim)                   # λLcon,θ,π(t(x)adv, {t′(x)}, {t(x)neg})

        return x.detach(), loss