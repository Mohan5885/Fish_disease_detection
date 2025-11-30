import os
from PIL import Image
from torch.utils.data import Dataset

class FishDataset(Dataset):
    """
    Expects:
    root_dir/
      train/
        classA/
        classB/
      val/
        classA/
        classB/
    """

    def __init__(self, root_dir, split='train', transform=None):
        split_dir = os.path.join(root_dir, split)
        if not os.path.isdir(split_dir):
            raise ValueError(f"Split directory not found: {split_dir}")
        classes = sorted([d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d))])
        if not classes:
            raise ValueError(f"No class folders found in {split_dir}")
        self.classes = classes
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.images = []
        self.labels = []
        for cls in self.classes:
            cls_dir = os.path.join(split_dir, cls)
            for fname in os.listdir(cls_dir):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.images.append(os.path.join(cls_dir, fname))
                    self.labels.append(self.class_to_idx[cls])
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        path = self.images[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        label = self.labels[idx]
        return img, label
