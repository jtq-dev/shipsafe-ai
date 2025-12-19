import os
import mlflow
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from app.ml.model import SmallCNN

os.makedirs("artifacts", exist_ok=True)
os.makedirs("artifacts/models", exist_ok=True)

def accuracy(logits, y):
    preds = logits.argmax(dim=1)
    return (preds == y).float().mean().item()

def main():
    mlflow.set_experiment("shipsafe-ai-mnist")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tfm = transforms.Compose([transforms.ToTensor()])

    train_ds = datasets.MNIST(root="artifacts/data", train=True, download=True, transform=tfm)
    test_ds  = datasets.MNIST(root="artifacts/data", train=False, download=True, transform=tfm)

    train_dl = DataLoader(train_ds, batch_size=128, shuffle=True)
    test_dl  = DataLoader(test_ds, batch_size=256)

    model = SmallCNN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()

    with mlflow.start_run():
        mlflow.log_param("model", "SmallCNN")
        mlflow.log_param("batch_size", 128)
        mlflow.log_param("lr", 1e-3)
        mlflow.log_param("device", device)

        for epoch in range(3):
            model.train()
            total_loss = 0.0
            for x, y in train_dl:
                x, y = x.to(device), y.to(device)
                opt.zero_grad()
                logits = model(x)
                loss = loss_fn(logits, y)
                loss.backward()
                opt.step()
                total_loss += loss.item()

            model.eval()
            with torch.no_grad():
                accs = []
                for x, y in test_dl:
                    x, y = x.to(device), y.to(device)
                    logits = model(x)
                    accs.append(accuracy(logits, y))

            avg_loss = total_loss / len(train_dl)
            avg_acc = sum(accs) / len(accs)

            mlflow.log_metric("train_loss", avg_loss, step=epoch)
            mlflow.log_metric("test_acc", avg_acc, step=epoch)
            print(f"epoch={epoch} loss={avg_loss:.4f} acc={avg_acc:.4f}")

        # Export TorchScript (trace)
        model_cpu = model.to("cpu").eval()
        example = torch.randn(1, 1, 28, 28)
        scripted = torch.jit.trace(model_cpu, example)

        out_path = "artifacts/models/mnist_scripted.pt"
        scripted.save(out_path)
        mlflow.log_artifact(out_path, artifact_path="model")

        print("✅ saved TorchScript:", out_path)
        print("✅ MLflow run logged (start UI with: mlflow ui)")

if __name__ == "__main__":
    main()
