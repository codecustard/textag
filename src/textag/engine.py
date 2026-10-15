"""
Training and validation loops for a PyTorch classifier.

train_step: one pass over the training data (forward pass, loss, backward pass, optimizer step)
val_step:   one pass over the validation data, no weight updates
train:      runs both every epoch, with optional
  - LR scheduler: lower the learning rate when validation loss stops improving
  - Early stopping: stop when validation loss stops improving, keep the best weights
"""
import copy

import torch
from tqdm.auto import tqdm
from typing import Dict, List, Tuple


def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device: torch.device) -> Tuple[float, float]:
    """Trains a model for one epoch. Returns (train_loss, train_acc)."""
    # Put model in train mode
    model.train()

    # Setup train loss and train accuracy values
    train_loss, train_acc = 0, 0

    # Loop through data loader data batches
    for batch, (X, y) in enumerate(dataloader):
        # Send data to target device
        X, y = X.to(device), y.to(device)

        # 1. Forward pass
        y_pred = model(X)

        # 2. Calculate and accumulate loss
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

        # Calculate and accumulate accuracy metric across all batches
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item() / len(y_pred)

    # Adjust metrics to get average loss and accuracy per batch
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc


def val_step(model: torch.nn.Module,
             dataloader: torch.utils.data.DataLoader,
             loss_fn: torch.nn.Module,
             device: torch.device) -> Tuple[float, float]:
    """Evaluates a model on the validation data for one epoch. Returns (val_loss, val_acc)."""
    # Put model in eval mode
    model.eval()

    # Setup val loss and val accuracy values
    val_loss, val_acc = 0, 0

    # Turn on inference context manager
    with torch.inference_mode():
        # Loop through DataLoader batches
        for batch, (X, y) in enumerate(dataloader):
            # Send data to target device
            X, y = X.to(device), y.to(device)

            # 1. Forward pass
            val_pred_logits = model(X)

            # 2. Calculate and accumulate loss
            loss = loss_fn(val_pred_logits, y)
            val_loss += loss.item()

            # Calculate and accumulate accuracy
            val_pred_labels = val_pred_logits.argmax(dim=1)
            val_acc += (val_pred_labels == y).sum().item() / len(val_pred_labels)

    # Adjust metrics to get average loss and accuracy per batch
    val_loss = val_loss / len(dataloader)
    val_acc = val_acc / len(dataloader)
    return val_loss, val_acc


def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          val_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int,
          device: torch.device,
          scheduler=None,
          patience: int = None,
          verbose: bool = True) -> Dict[str, List]:
    """Trains and validates a model for up to `epochs` epochs.

    scheduler: optional ReduceLROnPlateau, stepped on val loss each epoch.
    patience: optional early stopping; stop after this many epochs without val-loss improvement.
    verbose: print progress each epoch (set False to train quietly, e.g. for many quick runs).

    Returns a results dictionary, e.g. {train_loss: [...], train_acc: [...],
    val_loss: [...], val_acc: [...], best_epoch: 12}.
    The model ends up with the weights from the best epoch.
    """
    # Create empty results dictionary
    results = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    # Early stopping bookkeeping
    best_val_loss = float("inf")
    best_weights = None
    best_epoch = 0
    epochs_without_improvement = 0

    # Make sure model on target device
    model.to(device)

    # Loop through training and validation steps for a number of epochs
    for epoch in tqdm(range(epochs), disable=not verbose):
        train_loss, train_acc = train_step(model=model,
                                           dataloader=train_dataloader,
                                           loss_fn=loss_fn,
                                           optimizer=optimizer,
                                           device=device)
        val_loss, val_acc = val_step(model=model,
                                     dataloader=val_dataloader,
                                     loss_fn=loss_fn,
                                     device=device)

        # Print out what's happening
        if verbose:
            print(
                f"Epoch: {epoch+1} | "
                f"train_loss: {train_loss:.4f} | "
                f"train_acc: {train_acc:.4f} | "
                f"val_loss: {val_loss:.4f} | "
                f"val_acc: {val_acc:.4f} | "
                f"lr: {optimizer.param_groups[0]['lr']:.0e}"
            )

        # Update results dictionary
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["val_loss"].append(val_loss)
        results["val_acc"].append(val_acc)

        # LR scheduler: Halves the learning rate if val loss hasn't improved for a few epochs
        if scheduler is not None:
            scheduler.step(val_loss)

        # Early stopping: Remember the best weights; stop if no improvement for `patience` epochs
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if patience is not None and epochs_without_improvement >= patience:
                if verbose:
                    print(f"Early stopping: no val_loss improvement for {patience} epochs")
                break

    # Put the best weights back into the model
    model.load_state_dict(best_weights)
    results["best_epoch"] = best_epoch
    if verbose:
        print(f"Best epoch: {best_epoch} | val_loss: {best_val_loss:.4f} | "
              f"val_acc: {results['val_acc'][best_epoch - 1]:.4f}")

    # Return the filled results at the end of the epochs
    return results
