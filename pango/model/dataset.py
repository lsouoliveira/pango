from torch.utils.data import Dataset
import numpy as np

IMAGE_SIZE = 64


class CustomDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

        labels_index = df["label"].unique()
        labels_index = sorted(labels_index)

        self.label_to_index = {label: idx for idx, label in enumerate(labels_index)}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        label = row["label"]
        image_data = (
            row.drop(labels=["label"])
            .values.reshape(IMAGE_SIZE, IMAGE_SIZE)
            .astype(np.float32)
        )

        if self.transform:
            image_data = self.transform(image_data)

        return (image_data, self.label_to_index[label])
