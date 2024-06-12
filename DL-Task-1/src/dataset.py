import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class ImageDataset(Dataset):
    def __init__(self, txt_file, root_dir, channels="RGB", transform=None):
        """
        Args:
            txt_file (string): Path to the txt file with annotations.
            root_dir (string): Directory with all the images.
            channels (string): Desired channels to be used ("RGB", "RG", "GB", "R", "G", "B").
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.annotations = self._load_annotations(txt_file)
        self.root_dir = root_dir
        self.channels = channels
        self.transform = transform if transform else transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        self._print_sample_info() # 印出部分圖像的資訊

    def _load_annotations(self, txt_file):
        annotations = []
        with open(txt_file, 'r') as file:
            for line in file:
                img_path, label = line.strip().split()
                annotations.append((img_path, int(label)))
        return annotations

    def _select_channels(self, image):
        """
        Selects the specified channels from the image.
        """
        channel_map = {"R": 0, "G": 1, "B": 2}
        selected_channels = [channel_map[channel] for channel in self.channels if channel in channel_map]
        if len(selected_channels) == 1: # If only one channel is selected, repeat it to create a 3-channel image
            selected_image = image[selected_channels, :, :].repeat(3, 1, 1) 
        elif len(selected_channels) == 2: # If two channels are selected, add an empty channel
            selected_image = torch.zeros(3, image.size(1), image.size(2))
            selected_image[:2] = image[selected_channels, :, :]
        else: # Use the specified channels directly
            selected_image = image[selected_channels, :, :]
        return selected_image

    def _print_sample_info(self):
        for i in range(min(5, len(self.annotations))):  # 印出前5張圖片的資訊
            img_path, label = self.annotations[i]
            img_path = os.path.join(self.root_dir, img_path)
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            # print(f"Sample {i + 1}: Shape={image.shape}, Label={label}")

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        """
        Generates one sample of data.
        """
        img_path, label = self.annotations[idx]
        img_path = os.path.join(self.root_dir, img_path)
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)
        image = self._select_channels(image)  # 選擇指定的通道

        return image, label

# Example usage
if __name__ == "__main__":
    dataset = ImageDataset(txt_file='data/train.txt', root_dir='data/images')
    print(f"Number of samples: {len(dataset)}")
    sample_image, sample_label = dataset[0]
    print(f"Sample image shape: {sample_image.shape}, Sample label: {sample_label}")
