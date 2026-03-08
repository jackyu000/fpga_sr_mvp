import os  # Built-in Python module for interacting with the operating system

import torch  # Main PyTorch package
import torch.nn as nn  # Neural network utilities like loss functions
from torch.utils.data import DataLoader  # Helps batch and shuffle dataset samples

from tqdm import tqdm  # Nice progress bar for training loops

from dataset import CIFARGrayscaleSRDataset  # Our custom dataset class
from model import TinySRNet  # Our tiny super-resolution model


def psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> float:
    """
    Compute Peak Signal-to-Noise Ratio (PSNR).

    PSNR is a common metric for image reconstruction quality.
    Higher PSNR generally means the prediction is closer to the target.

    pred:   model output image tensor
    target: ground-truth target image tensor
    max_val: maximum valid pixel value; since our tensors are in [0,1], max_val=1.0
    """

    # Compute Mean Squared Error (MSE) between prediction and target
    # (pred - target) gives pixelwise error
    # ** 2 squares each error
    # torch.mean(...) averages over all pixels in the tensor
    # .item() converts a scalar tensor to a normal Python float
    mse = torch.mean((pred - target) ** 2).item()

    # If MSE is exactly zero, the images are identical
    # PSNR would mathematically be infinite, but we return a large number instead
    if mse == 0:
        return 99.0

    # Import math here for log10
    import math

    # PSNR formula:
    # PSNR = 20 * log10(MAX / sqrt(MSE))
    return 20 * math.log10(max_val / (mse ** 0.5))


def evaluate(model, loader, device):
    """
    Run the model on the validation/test set and compute average metrics.

    model: the neural network
    loader: DataLoader providing batches from validation/test data
    device: CPU or GPU

    Returns:
        average validation loss,
        average PSNR of model output,
        average PSNR of bicubic baseline
    """

    # Switch model into evaluation mode
    # Important for layers like BatchNorm or Dropout
    # Our model doesn't use those, but this is still correct practice
    model.eval()

    # Running sums for metrics
    total_loss = 0.0
    total_psnr_model = 0.0
    total_psnr_bicubic = 0.0

    # L1 loss = mean absolute error
    # This compares model output to ground truth
    criterion = nn.L1Loss()

    # Disable gradient tracking during evaluation
    # This saves memory and computation because we are not training here
    with torch.no_grad():

        # Iterate through validation/test batches
        # Each batch from our dataset is:
        #   lr_32, bicubic_64, hr_64
        # We ignore lr_32 here using "_"
        for _, bicubic_64, hr_64 in loader:

            # Move tensors to CPU or GPU depending on chosen device
            bicubic_64 = bicubic_64.to(device)
            hr_64 = hr_64.to(device)

            # Run forward pass through the model
            pred = model(bicubic_64)

            # Compute L1 loss between prediction and target
            loss = criterion(pred, hr_64)

            # Add this batch's loss to running total
            total_loss += loss.item()

            # Compute PSNR for model output vs target
            total_psnr_model += psnr(pred, hr_64)

            # Compute PSNR for raw bicubic input vs target
            # This tells us whether the model is actually improving over bicubic
            total_psnr_bicubic += psnr(bicubic_64, hr_64)

    # Number of batches in loader
    n = len(loader)

    # Return average metrics across all batches
    return (
        total_loss / n,
        total_psnr_model / n,
        total_psnr_bicubic / n,
    )


def main():
    """
    Main training function.

    This sets up:
    - device
    - datasets
    - dataloaders
    - model
    - optimizer
    - loss function
    - training loop
    - validation loop
    - checkpoint saving
    """

    # Choose device:
    # If CUDA GPU is available, use it
    # otherwise use CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create training dataset (CIFAR10 train split)
    train_ds = CIFARGrayscaleSRDataset(train=True)

    # Create validation/test dataset (CIFAR10 test split)
    test_ds = CIFARGrayscaleSRDataset(train=False)

    # DataLoader wraps the dataset and provides batches
    #
    # batch_size=64 means each batch has 64 images
    # shuffle=True randomizes order each epoch, which helps training
    # num_workers=2 means 2 background worker processes load data in parallel
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=2)

    # For test/validation, shuffle=False because order doesn't matter
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=2)

    # Create model instance with 8 internal feature channels
    # .to(device) moves model parameters to CPU or GPU
    model = TinySRNet(channels=8).to(device)

    # Adam optimizer updates model weights using computed gradients
    #
    # model.parameters() tells Adam which trainable weights to update
    # lr=1e-3 is learning rate = step size for weight updates
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # L1 loss = mean absolute difference between output and target
    # Often works well for image restoration tasks
    criterion = nn.L1Loss()

    # Ensure checkpoints folder exists
    # exist_ok=True means don't error if folder is already there
    os.makedirs("checkpoints", exist_ok=True)

    # Number of passes through the full training dataset
    num_epochs = 10

    # Outer loop: one epoch = one full pass through training set
    for epoch in range(num_epochs):

        # Switch model to training mode
        # Important if using dropout/batchnorm
        model.train()

        # Accumulate loss over all batches in this epoch
        running_loss = 0.0

        # tqdm adds a progress bar around the training DataLoader
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

        # Iterate through batches
        # Our dataset returns (lr_32, bicubic_64, hr_64)
        # We only need bicubic_64 and hr_64 for this training loop
        for _, bicubic_64, hr_64 in pbar:

            # Move current batch tensors to device
            bicubic_64 = bicubic_64.to(device)
            hr_64 = hr_64.to(device)

            # Forward pass:
            # feed bicubic image into model to get predicted refined image
            pred = model(bicubic_64)

            # Compare prediction to ground truth HR target
            loss = criterion(pred, hr_64)

            # Zero out old gradients before computing new ones
            # Gradients accumulate by default in PyTorch, so this is required
            optimizer.zero_grad()

            # Backpropagation:
            # compute gradients of loss with respect to all model parameters
            loss.backward()

            # Optimizer step:
            # use gradients to update weights
            optimizer.step()

            # Add current batch loss to total epoch loss
            running_loss += loss.item()

            # Update progress bar display with current batch loss
            pbar.set_postfix(loss=loss.item())

        # Average training loss over all batches in this epoch
        train_loss = running_loss / len(train_loader)

        # Evaluate on validation/test set after each epoch
        val_loss, val_psnr_model, val_psnr_bicubic = evaluate(model, test_loader, device)

        # Print summary for this epoch
        print(
            f"Epoch {epoch+1}: "
            f"train_loss={train_loss:.4f}, "
            f"val_loss={val_loss:.4f}, "
            f"PSNR(model)={val_psnr_model:.2f}, "
            f"PSNR(bicubic)={val_psnr_bicubic:.2f}"
        )

        # Save checkpoint after every epoch
        # state_dict() contains all learned weights and biases
        torch.save(model.state_dict(), f"checkpoints/tinysr_epoch_{epoch+1}.pt")

    # Save final model after training finishes
    torch.save(model.state_dict(), "checkpoints/tinysr_final.pt")

    print("Training complete. Saved model to checkpoints/tinysr_final.pt")


# Standard Python idiom:
# only run main() if this file is executed directly
# not when imported as a module from another file
if __name__ == "__main__":
    main()