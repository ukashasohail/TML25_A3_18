# TML Assignment 3: Robustness

This project implements a defense mechanism against adversarial attacks (FGSM and PGD) by enhancing the robustness of an image encoder using adversarial training. The training strategy is inspired by recent works in adversarial self-supervised learning and saddle-point optimization.

## 📌 Objective

To improve the robustness of a ResNet18 encoder against adversarial perturbations using a combination of:
- **Contrastive loss** in the inner maximization loop (attack phase)
- **Cross-entropy loss** in the outer minimization loop (defense/training phase)

The methodology follows the **saddle-point optimization** framework proposed by:
- Kim et al. in *Adversarial Self-Supervised Contrastive Learning*
- Madry et al. in *Towards Deep Learning Models Resistant to Adversarial Attacks*

---

## 🧠 Approach: Solving the Saddle-Point Problem

We optimized the following formulation:

> **LRoCL<sub>θ,π</sub> + λL<sub>con,θ,π</sub>(t(x)<sub>adv</sub>, {t′(x)}, {t(x)<sub>neg</sub>})**

Where:
- `LRoCL` is the cross-entropy loss (outer minimization)
- `Lcon` is the contrastive loss (inner maximization)
- `λ` is a scaling factor (set to `2/255`)

---

## 🧪 PGD Formulation (Inner Maximization)

1. **Stochastic Augmentations**:
   - Applied two augmentations per image: `t(x)` and `t′(x)`
   - Used: `RandAugment`, `ColorJitter`, `RandomGrayscale`, `HorizontalFlip`

2. **PGD Attack Steps**:
   - Started with random noise within `ε = 8/255`
   - Iteratively optimized negative contrastive loss (7 steps, α = 2/255)
   - Used **NT-Xent loss** between `t(x)_adv` and `t′(x)` (temperature = 0.5) with in-batch negatives
   - Projected adversarial image back into the L-infinity norm ball

---

## 🏗️ Robust Encoder (Custom ResNet18)

- **Backbone**: ResNet18
- **Projector Head**: 128-Dimensional MLP to project embeddings into a latent space
- **Classifier Head**: 10 output dimensions for CIFAR-10
- The encoder outputs both **logits** and **latent representations** for hybrid loss computation.

---

## 🏋️ Adversarial Training Loop (Outer Minimization)

1. Generate `t(x)` and `t′(x)`
2. Generate adversarial embedding from `t(x)`
3. Compute:
   - **Cross-entropy loss** on classifier output
   - **Contrastive loss** is obtained from the PGD implementation
4. Combine both losses to optimize saddle-point objective
5. Fine-tuned the encoder (projector head removed) before evaluation

---

## ✅ Performance Evaluation

| Metric           | Accuracy |
|------------------|----------|
| Clean Accuracy   | 0.597    |
| FGSM Accuracy    | 0.266    |
| PGD Accuracy     | 0.001    |

- Evaluation performed using both clean test data and perturbed test data via PGD/FGSM attacks.

---

## 🗂️ Code Structure

| File                  | Description                                                                 |
|-----------------------|-----------------------------------------------------------------------------|
| `RepresentationAdversarial.py` | Crafts adversarial examples and performs L-infinity projection      |
| `loss.py`             | Computes pairwise similarity and NT-Xent contrastive loss                  |
| `RobustEncoder.py`    | Defines the custom ResNet18 with projector and classifier heads            |
| `Dataset.py`          | Generates two stochastic views (`t(x)`, `t′(x)`) for contrastive learning  |
| `Evaluation.py`       | Computes clean, FGSM, and PGD accuracy                                     |
| `Train.py`            | Training loop using hybrid loss (contrastive + cross-entropy)              |
| `Config.py`           | Centralized configuration of hyperparameters                              |
| `Main.py`             | Entry point for training and evaluation                                    |

---

## 🔗 Repositories and Online Resources Consulted

- [RoCL – Adversarial Self-Supervised Contrastive Learning](https://github.com/Kim-Minseon/RoCL)
- [CIFAR10 Adversarial Examples Challenge (MadryLab)](https://github.com/MadryLab/cifar10_challenge/tree/master)
- [NT-Xent Loss Explanation – Medium Article](https://medium.com/self-supervised-learning/nt-xent-loss-normalized-temperature-scaled-cross-entropy-loss-ea5a1ede7c40)

---

## 📚 References

1. Kim, M., Tack, J., & Hwang, S. J. (n.d.). *Adversarial Self-Supervised Contrastive Learning*. KAIST, AITRICS. https://arxiv.org/abs/2006.07589  
2. Madry, A., Makelov, A., Schmidt, L., Tsipras, D., & Vladu, A. (2018). *Towards Deep Learning Models Resistant to Adversarial Attacks*. ICLR. https://arxiv.org/abs/1706.06083  
3. Adversarial ML Tutorial. (n.d.). *Adversarial Training, Solving the Outer Minimization*. https://adversarial-ml-tutorial.org/adversarial_training/  
4. Papernot, N., McDaniel, P., Goodfellow, I., Jha, S., Celik, Z. B., & Swami, A. (2017). *Practical Black-Box Attacks Against Machine Learning*. ACM Asia CCS. https://doi.org/10.1145/3052973.3053009

---

## 📎 GitHub Repository

🔗 [Project Repository](https://github.com/ukashasohail/TML25_A3_18)
