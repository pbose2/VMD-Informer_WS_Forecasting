import torch
import torch.nn as nn
import matplotlib.pyplot as plt


def train_model(model, train_loader, val_loader, lr, epochs, save_path, device):
    """Train model, save best checkpoint, return (model, history)."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    criterion = nn.MSELoss()

    best_val = float('inf')
    history  = {'train': [], 'val': []}

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                val_loss += criterion(model(xb), yb).item()
        val_loss /= len(val_loader)

        scheduler.step(val_loss)
        history['train'].append(train_loss)
        history['val'].append(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), save_path)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d}/{epochs} | train {train_loss:.4f} | val {val_loss:.4f} | best {best_val:.4f}")

    print(f"\nTraining done — best val loss: {best_val:.4f}")
    model.load_state_dict(torch.load(save_path))
    return model, history


def plot_loss(history, title):
    plt.figure(figsize=(10, 4))
    plt.plot(history['train'], label='Train')
    plt.plot(history['val'],   label='Val')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()
