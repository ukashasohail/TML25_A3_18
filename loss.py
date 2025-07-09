import torch

# [5.0]
def pairwise_similarity(outputs, temperature=0.5):
    """
        Compute pairwise similarity and return the matrix
        input: aggregated outputs & temperature for scaling
        return: pairwise cosine similarity
    """
    norm = outputs / (outputs.norm(dim=1, keepdim=True) + 1e-8)
    return (1. / temperature) * torch.mm(norm, norm.T.detach()), outputs

def NT_xent(similarity_matrix):
    """
        Compute NT_xent loss
        input: pairwise-similarity matrix
        return: NT xent loss
    """
    N2 = similarity_matrix.size(0)
    N = N2 // 2
    sim_exp = torch.exp(similarity_matrix) * (1 - torch.eye(N2, device=similarity_matrix.device))
    loss_matrix = -torch.log(sim_exp / (sim_exp.sum(dim=1, keepdim=True) + 1e-8) + 1e-8)

    loss = loss_matrix.diag(N).sum() + loss_matrix.diag(-N).sum()

    return loss / N2