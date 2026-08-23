import os
import time
import torch
import torch.nn as nn
from tqdm import tqdm


class EarlyStopping:
    def __init__(self, patience: int = 5, min_delta: float = 0.001, save_path: str = "best_model.pth"):
        self.patience = patience
        self.min_delta = min_delta
        self.save_path = save_path
        self.counter = 0
        self.best_loss = float("inf")
        self.triggered = False

    def step(self, val_loss: float, model: nn.Module) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            torch.save(model.state_dict(), self.save_path)
            return False
        self.counter += 1
        if self.counter >= self.patience:
            self.triggered = True
            return True
        return False


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in tqdm(loader, desc="  train", leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
    return total_loss / len(loader), 100.0 * correct / total


def validate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="  val  ", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    return total_loss / len(loader), 100.0 * correct / total


def train(model, train_loader, val_loader, cfg: dict, save_dir: str, run_name: str):
    """
    Full training loop driven by config dict. Returns history dict with
    train_losses, val_losses, train_accs, val_accs.
    """
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Training on {device}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg["learning_rate"],
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=cfg["lr_scheduler"]["step_size"],
        gamma=cfg["lr_scheduler"]["gamma"],
    )
    early_stopping = EarlyStopping(
        patience=cfg["early_stopping_patience"],
        min_delta=cfg["min_delta"],
        save_path=os.path.join(save_dir, f"best_{run_name}.pth"),
    )

    history = {"train_losses": [], "val_losses": [], "train_accs": [], "val_accs": []}
    best_val_acc = 0.0

    for epoch in range(cfg["num_epochs"]):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc     = validate(model, val_loader, criterion, device)
        scheduler.step()

        history["train_losses"].append(train_loss)
        history["val_losses"].append(val_loss)
        history["train_accs"].append(train_acc)
        history["val_accs"].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc

        print(
            f"Epoch [{epoch+1:02d}/{cfg['num_epochs']}] {time.time()-t0:.1f}s  "
            f"train loss {train_loss:.4f} acc {train_acc:.2f}%  |  "
            f"val loss {val_loss:.4f} acc {val_acc:.2f}%"
        )

        if early_stopping.step(val_loss, model):
            print(f"Early stopping at epoch {epoch+1}. Best val loss: {early_stopping.best_loss:.4f}")
            break

    torch.save(model.state_dict(), os.path.join(save_dir, f"final_{run_name}.pth"))
    print(f"Best val accuracy: {best_val_acc:.2f}%")
    return history
