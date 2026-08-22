import matplotlib

matplotlib.use("Agg")  # runs headless; the plot goes to a file, not a window
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader

from data import split, augment
from model import model, criterion, optimizer

batch_size = 64
epochs = 75
patience = 15  
seed = 42

torch.manual_seed(seed)

train_loader = DataLoader(split("train"), batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(split("valid"), batch_size=256)
test_loader = DataLoader(split("test"), batch_size=256)

best_acc = 0.0
stale = 0
history = []
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

    
    if valid_acc > best_acc:
        best_acc = valid_acc
        stale = 0
        torch.save(model.state_dict(), "model.pt")
    else:
        stale += 1

    history.append((train_loss, valid_loss, valid_acc))
    print(f"epoch {epoch:2d}  train loss {train_loss:.4f}  val loss {valid_loss:.4f}  val acc {valid_acc:.4f}")

    if stale >= patience:
        print(f"no val improvement in {patience} epochs, stopping at epoch {epoch}")
        break

model.load_state_dict(torch.load("model.pt"))
model.eval()
correct = 0
test_loss = 0.0
with torch.no_grad():
    for x, y in test_loader:
        out = model(x)
        test_loss += criterion(out, y).item() * len(x)
        correct += (out.argmax(1) == y).sum().item()
test_loss /= len(test_loader.dataset)
test_acc = correct / len(test_loader.dataset)

print("Best validation accuracy:", round(best_acc, 4))
print("Test loss:", round(test_loss, 4))
print("Test accuracy:", round(test_acc, 4))

# train loss is on augmented batches, val loss on clean images - the two curves
# are not directly comparable, only their trends are
train_losses, valid_losses, valid_accs = zip(*history)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
ax1.plot(range(1, len(history) + 1), train_losses, label="train (augmented)")
ax1.plot(range(1, len(history) + 1), valid_losses, label="valid (clean)")
ax1.set_xlabel("epoch")
ax1.set_ylabel("loss")
ax1.legend()
ax2.plot(range(1, len(history) + 1), valid_accs, color="tab:green")
ax2.axhline(best_acc, ls="--", lw=0.8, color="gray")
ax2.set_xlabel("epoch")
ax2.set_ylabel("valid accuracy")
fig.tight_layout()
fig.savefig("curves.png", dpi=130)
print("wrote curves.png")


y_pred = []
y_true = []
with torch.no_grad():
    for x, y in test_loader:
        y_pred.append(model(x).argmax(1))
        y_true.append(y)
y_pred = torch.cat(y_pred).numpy()
y_true = torch.cat(y_true).numpy()

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(14, 12))

sns.heatmap(cm, cmap="viridis", square=True, cbar_kws={"shrink": 0.6})
plt.title("Confusion Matrix")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.savefig("confusion.png", dpi=130, bbox_inches="tight")
print("wrote confusion.png")
