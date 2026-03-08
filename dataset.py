import torch

# Base class used for creating custom datasets
from torch.utils.data import Dataset

# torchvision provides standard datasets and image transforms
from torchvision import datasets, transforms

# PIL provides image operations like resizing
from PIL import Image


class CIFARGrayscaleSRDataset(Dataset):
    """
    Dataset for super-resolution training.

    For each image we generate three versions:

        lr_32       -> low-resolution image (32x32)
        bicubic_64  -> blurred upscaled image (model input)
        hr_64       -> high-resolution target

    Shapes returned as tensors:

        lr_32:      (1, 32, 32)
        bicubic_64: (1, 64, 64)
        hr_64:      (1, 64, 64)
    """

    def __init__(self, train: bool = True):
        """
        Initializes the dataset.

        train=True loads the training split of CIFAR10.
        train=False loads the test split.
        """

        # Load CIFAR10 dataset
        # root="./data" stores dataset locally in project folder
        # download=True automatically downloads if missing
        self.dataset = datasets.CIFAR10(
            root="./data",
            train=train,
            download=True,
        )

        # Transformation that converts RGB image -> grayscale
        # num_output_channels=1 ensures single channel output
        self.to_gray = transforms.Grayscale(num_output_channels=1)

        # Converts PIL image -> PyTorch tensor
        # Also rescales pixel values from [0,255] -> [0,1]
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        """
        Returns number of samples in dataset.

        Required for PyTorch Dataset interface.
        """

        return len(self.dataset)

    def __getitem__(self, idx):
        """
        Returns a single training example.

        idx = index of the sample in the dataset
        """

        # dataset[idx] returns:
        #   (image, label)
        # We only care about the image so we discard the label with "_"
        img, _ = self.dataset[idx]

        # Convert RGB image to grayscale
        # Result is a PIL image of size 32x32
        gray_32 = self.to_gray(img)

        # Create "high resolution" target
        # Upsample grayscale image to 64x64 using bicubic interpolation
        hr_64 = gray_32.resize((64, 64), Image.BICUBIC)

        # Create degraded low-resolution image
        # Downsample back to 32x32
        lr_32 = hr_64.resize((32, 32), Image.BICUBIC)

        # Upsample again to 64x64
        # This creates a blurry bicubic baseline image
        bicubic_64 = lr_32.resize((64, 64), Image.BICUBIC)

        # Convert all images to PyTorch tensors
        # Resulting shapes:
        #   lr_32_t      -> (1, 32, 32)
        #   bicubic_64_t -> (1, 64, 64)
        #   hr_64_t      -> (1, 64, 64)
        lr_32_t = self.to_tensor(lr_32)
        bicubic_64_t = self.to_tensor(bicubic_64)
        hr_64_t = self.to_tensor(hr_64)

        # Return tuple used during training
        # Model input = bicubic_64
        # Ground truth = hr_64
        return lr_32_t, bicubic_64_t, hr_64_t