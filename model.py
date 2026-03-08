import torch                      # Main PyTorch package (tensors, math ops, device handling)
import torch.nn as nn             # Neural network module containing layers like Conv2d, ReLU, etc.


class TinySRNet(nn.Module):
    """
    Tiny Super-Resolution Network.

    This network takes a blurry bicubic-upscaled image and learns a small
    correction (called a residual) that sharpens the image.

    Instead of predicting the whole high-resolution image from scratch,
    the model predicts only the correction to add to the bicubic input.

    Input tensor shape:
        (B, 1, 64, 64)
        B = batch size
        1 = grayscale channel
        64x64 = spatial dimensions

    Output tensor shape:
        (B, 1, 64, 64)
    """

    def __init__(self, channels: int = 8):
        """
        Constructor for the model.

        channels controls how many internal feature maps we use.
        8 is intentionally small so the model is lightweight and FPGA-friendly.
        """

        super().__init__()  # Initialize the base PyTorch module

        # nn.Sequential creates a pipeline of layers executed in order
        # input -> layer1 -> layer2 -> layer3 -> ...
        self.net = nn.Sequential(

            # First convolution layer
            # Takes 1 input channel (grayscale image)
            # Produces 'channels' feature maps (default = 8)
            # kernel_size=3 means a 3x3 filter
            # padding=1 preserves spatial size (64x64 stays 64x64)
            nn.Conv2d(1, channels, kernel_size=3, padding=1),

            # ReLU activation
            # Applies max(0, x) elementwise
            # Introduces nonlinearity so the network can learn complex functions
            nn.ReLU(inplace=True),

            # Second convolution
            # Takes the 8 feature maps and produces 8 new ones
            # These feature maps represent learned image features
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),

            # Another ReLU activation
            nn.ReLU(inplace=True),

            # Third convolution
            # Again maps 8 feature maps -> 8 feature maps
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),

            # ReLU activation again
            nn.ReLU(inplace=True),

            # Final convolution
            # Maps the internal feature maps back to a single channel
            # This output represents the learned "residual correction"
            nn.Conv2d(channels, 1, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the network.

        This defines how an input image flows through the model.
        """

        # Pass the input through the convolution stack
        # This produces the learned residual image
        residual = self.net(x)

        # Add the residual correction to the input image
        # This is called residual learning
        out = x + residual

        # Clamp ensures the output stays in valid pixel range [0,1]
        # because tensors from ToTensor() are normalized to that range
        return torch.clamp(out, 0.0, 1.0)