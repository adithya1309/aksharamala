import torch
from torch.utils.data import DataLoader

from data import split, augment
from model import model, criterion, optimizer

batch_size = 64
epochs = 50
patience = 5
seed = 42

torch.manual_seed(seed)

train_loader = DataLoader(split("train"), batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(split("valid"), batch_size=256)
test_loader = DataLoader(split("test"), batch_size=256)

best_acc = 0.0
stale = 0
for epoch in range(1, epochs + 1):
    model.train()
    train_loss = 0.0
    for x, y in train_loader:
        x = augment(x)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * len(x)
    train_loss /= len(train_loader.dataset)

    model.eval()
    correct = 0
    valid_loss = 0.0
    with torch.no_grad():
        for x, y in valid_loader:
            out = model(x)
            valid_loss += criterion(out, y).item() * len(x)
            correct += (out.argmax(1) == y).sum().item()
    valid_loss /= len(valid_loader.dataset)
    valid_acc = correct / len(valid_loader.dataset)

    # keep the best-on-validation weights, not whatever the last epoch happens to be
    if valid_acc > best_acc:
        best_acc = valid_acc
        stale = 0
        torch.save(model.state_dict(), "model.pt")
    else:
        stale += 1

    print(f"epoch {epoch:2d}  train loss {train_loss:.4f}  val loss {valid_loss:.4f}  val acc {valid_acc:.4f}")

    if stale >= patience:
        print(f"no val improvement in {patience} epochs, stopping at epoch {epoch}")
        break

# test only once, with the best checkpoint - the held-out writers are the honest number
model.load_state_dict(torch.load("model.pt"))
model.eval()
correct = 0
with torch.no_grad():
    for x, y in test_loader:
        correct += (model(x).argmax(1) == y).sum().item()
print(f"best val acc {best_acc:.4f}  test acc {correct / len(test_loader.dataset):.4f}")
