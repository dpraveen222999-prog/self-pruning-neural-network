import torch
import matplotlib.pyplot as plt


def compute_sparsity(model, threshold=1e-2):
    total, pruned = 0, 0
    layer_stats = []

    for m in model.modules():
        if hasattr(m, "gate_scores"):
            gates = torch.sigmoid(m.gate_scores).detach()
            total += gates.numel()
            pruned_layer = (gates < threshold).sum().item()
            pruned += pruned_layer
            layer_stats.append((pruned_layer / gates.numel()) * 100)

    overall = 100 * pruned / total if total > 0 else 0.0
    return overall, layer_stats


def get_all_gates(model):
    all_gates = []
    for m in model.modules():
        if hasattr(m, "gate_scores"):
            gates = torch.sigmoid(m.gate_scores).detach().cpu().view(-1)
            all_gates.append(gates)
    if len(all_gates) == 0:
        return torch.tensor([])
    return torch.cat(all_gates)


def plot_histogram(gates, save_path):
    if gates.numel() == 0:
        return
    plt.figure()
    plt.hist(gates.numpy(), bins=60)
    plt.title("Gate Value Distribution")
    plt.xlabel("Gate Value")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_layer_sparsity(layer_stats, save_path):
    if not layer_stats:
        return
    plt.figure()
    plt.bar(range(len(layer_stats)), layer_stats)
    plt.xlabel("Layer Index")
    plt.ylabel("Sparsity (%)")
    plt.title("Layer-wise Sparsity")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
