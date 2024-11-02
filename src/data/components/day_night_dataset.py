import os
import pandas as pd
import torch
from torchvision.transforms import v2
from torchvision.io import read_image


class DayNightDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None, shuffle=True):
        self.data_dir = data_dir
        self.data = pd.read_csv(os.path.join(data_dir, "train_20241023/train_data.csv"))
        if shuffle is True:
            self.data = self.data.sample(frac=1).reset_index(drop=True)
        if transform is not None:
            self.transform = transform
        else:
            self.transform = v2.Compose([
                v2.Resize([512, 512]),
                v2.RandomHorizontalFlip(p=0.3),
                v2.RandomVerticalFlip(p=0.3),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_dir, "train_20241023", self.data.iloc[idx, 0])
        image = read_image(img_path)
        label = self.data.iloc[idx, 1]
        image = self.transform(image)
        label = torch.tensor(label, dtype=torch.float32).unsqueeze(0)

        return image, label
    
if __name__ == "__main__":
    dataset = DayNightDataset("./data")
    image, label = dataset[0]
    print(image.shape)
    print(label.shape)
    print(image)
    print(label)