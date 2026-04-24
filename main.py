import argparse
import json
import os
import random
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from model import PrunableMLP
from utils import compute_sparsity, get_all_gates, plot_histogram, plot_layer_sparsity


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_data(batch_size):
    transform = transforms.Compose([transforms.ToTensor()])

    trainset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    testset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
    )

    return trainloader, testloader


def sparsity_loss(model, temperature):
    loss = 0.0
    for m in model.modules():
        if hasattr(m, "gate_scores"):
            loss = loss + torch.sum(torch.sigmoid(m.gate_scores / temperature))
    return loss


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        preds = model(x).argmax(1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    return 100.0 * correct / max(total, 1)


def train_one_lambda(lmbda, args):
    device = get_device()
    trainloader, testloader = get_data(args.batch_size)

    model = PrunableMLP(temperature=args.temperature).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    history = []
    layer_stats = []

    
    sp = 0

    if args.epochs > 0:
        for epoch in range(1, args.epochs + 1):
            model.train()
            t0 = time.time()
            running_loss = 0.0

            for x, y in trainloader:
                x, y = x.to(device), y.to(device)

                out = model(x)
                ce_loss = criterion(out, y)
                sp_loss = sparsity_loss(model, args.temperature)
                loss = ce_loss + lmbda * sp_loss

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            acc = evaluate(model, testloader, device)
            sp, layer_stats = compute_sparsity(model)

            history.append({
                "epoch": epoch,
                "loss": running_loss / max(len(trainloader), 1),
                "acc": acc,
                "sparsity": sp
            })

            if epoch % args.log_every == 0 or epoch == args.epochs:
                print(f"[λ={lmbda}] epoch {epoch:02d} | loss={history[-1]['loss']:.4f} | acc={acc:.2f} | sparsity={sp:.2f} | time={time.time()-t0:.1f}s")

            else:
                # Handle epochs = 0 case safely
                acc = evaluate(model, testloader, device)
                sp, layer_stats = compute_sparsity(model)

    

    # Artifacts
    gates = get_all_gates(model)
    out_dir = os.path.join(args.output_dir, f"lambda_{str(lmbda).replace('.', '_')}")
    os.makedirs(out_dir, exist_ok=True)

    plot_histogram(gates, os.path.join(out_dir, "gates_hist.png"))
    if layer_stats is None or len(layer_stats) == 0:
        _, layer_stats = compute_sparsity(model)
    plot_layer_sparsity(layer_stats, os.path.join(out_dir, "layer_sparsity.png"))

    # Save metrics
    if len(history) > 0:
        final_acc = history[-1]["acc"]
        final_sp = history[-1]["sparsity"]
    else:
        # epochs = 0 case
        final_acc = acc
        final_sp = sp

    metrics = {
        "lambda": lmbda,
        "final_accuracy": final_acc,
        "final_sparsity": final_sp,
        "epochs": args.epochs,
        "temperature": args.temperature
    }
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


def write_results_md(results, path):
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Results\n\n")
        f.write("| Lambda | Accuracy (%) | Sparsity (%) |\n")
        f.write("|--------|--------------|--------------|\n")
        for r in results:
            f.write(f"| {r['lambda']} | {r['final_accuracy']:.2f} | {r['final_sparsity']:.2f} |\n")

        f.write("\n---\n\n")
        f.write("## Interpretation\n\n")
        f.write("- λ = 0 establishes the baseline without pruning.\n")
        f.write("- Small λ introduces mild sparsity with minimal accuracy impact.\n")
        f.write("- Moderate λ typically yields the best trade-off.\n")
        f.write("- Large λ drives aggressive pruning, degrading accuracy due to capacity loss.\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--log_every", type=int, default=1)
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    lambdas = [0.0, 1e-4, 1e-3, 1e-2]
    results = []

    for l in lambdas:
        print(f"\n=== Training with λ={l} ===")
        metrics = train_one_lambda(l, args)
        results.append(metrics)

    write_results_md(results, os.path.join(args.output_dir, "results.md"))
    print("\nDone. See outputs/ for artifacts.")


if __name__ == "__main__":
    main()
