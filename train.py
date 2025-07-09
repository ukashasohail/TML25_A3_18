from RepresentationAdversarial import RepresentationAdversarial
from config import EPOCHS, LEARNING_RATE, device
from Evaluation import eval_clean, eval_fgsm, eval_pgd

import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

# [6.0]
def train(model, train_loader, test_loader, num_epochs=EPOCHS):
    adv_gen = RepresentationAdversarial(model)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    model.train()
    epoch_losses = []

    for epoch in range(num_epochs):
        total_loss, reg_loss, cls_loss = 0.0, 0.0, 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for original, view1, view2, _, labels in pbar:
            original, view1, view2 = original.to(device), view1.to(device), view2.to(device)
            labels = labels.to(device).repeat(3)

            adv_imgs, adv_loss = adv_gen.get_loss(view1, view2, optimizer)
            reg_loss += adv_loss.item()

            inputs = torch.cat((view1, view2, adv_imgs))
            z, logits = model(inputs)
            ce_loss = F.cross_entropy(logits, labels)
            loss = ce_loss + adv_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            cls_loss += ce_loss.item()

            pbar.set_postfix({
                'Loss': total_loss / (pbar.n + 1),
                'CE': cls_loss / (pbar.n + 1),
                'Adv': reg_loss / (pbar.n + 1)
            })

        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            ckpt_path = f"./checkpoints/robust_model_epoch_{epoch+1}.pth"
            torch.save(model.state_dict(), ckpt_path)
            print(f" Saved checkpoint to: {ckpt_path}")

        # Always track and print loss
        epoch_loss = total_loss / len(train_loader)
        epoch_losses.append(epoch_loss)
        print(f"Epoch {epoch+1} Loss: {epoch_loss:.4f}")

        # Evaluate clean and PGD accuracy
        clean_acc = eval_clean(model, test_loader)
        fgsm_acc = eval_fgsm(model, test_loader)
        pgd_acc = eval_pgd(model, test_loader)
        print(f" Clean Acc: {clean_acc:.4f} | FGSM Acc: {fgsm_acc:.4f} | PGD Acc: {pgd_acc:.4f}")

    torch.save(model.state_dict(), "./model/robust_model.pth")
    return epoch_losses