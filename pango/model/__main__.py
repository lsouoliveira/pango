import torch
import pandas as pd
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn as nn
import numpy as np

from .dataset import CustomDataset
from .model import Model

IMAGE_SIZE = 64

device = "cuda" if torch.cuda.is_available() else "cpu"


def print_labels_count(dataset):
    print(dataset["label"].value_counts())


def show_example_images(dataset, cols=5, rows=3):
    _, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    selected_indices = random.sample(range(len(dataset)), cols * rows)

    X = dataset.drop(columns=["label"])

    for ax, idx in zip(axes.flatten(), selected_indices):
        image_data = X.iloc[idx].values.reshape(IMAGE_SIZE, IMAGE_SIZE)
        label = dataset.iloc[idx]["label"]

        ax.imshow(image_data, cmap="gray")
        ax.set_title(f"Label: {label}")
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def show_distribution_of_labels(dataset):
    label_counts = dataset["label"].value_counts()
    label_counts.plot(kind="bar", figsize=(10, 6))
    plt.xlabel("Labels")
    plt.ylabel("Count")
    plt.title("Distribution of Labels")
    plt.show()


def show_data_analysis(data):
    print_labels_count(data)
    show_example_images(data)
    show_distribution_of_labels(data)


def drop_classes_with_few_samples(data, min_samples=2):
    counts = data["label"].value_counts()
    to_keep = counts[counts >= min_samples].index
    filtered_data = data[data["label"].isin(to_keep)].reset_index(drop=True)

    return filtered_data


def apply_data_augmentation(X, Y):
    return X, Y


def main():
    data = pd.read_csv("dataset.csv")
    data = drop_classes_with_few_samples(data, min_samples=2)

    train, test = train_test_split(data, test_size=0.1, random_state=42)

    show_data_analysis(train)

    train_transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x / 255.0),
        ]
    )

    train_dataset = CustomDataset(train, transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    imgs, lbls = next(iter(train_loader))

    print(f"Image batch shape: {imgs.size()}")
    print(f"Labels in batch: {lbls}")
    print(f"Min value: {imgs.min()}")
    print(f"Max value: {imgs.max()}")
    print(f"Mean: {imgs.mean()}")
    print(f"Std: {imgs.std()}")

    model = Model()
    model.to(device)

    print(f"Model will run on: {device}")

    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print(model)

    epochs = 100
    train_losses = []
    train_accuracies = []

    for epoch in range(epochs):
        running_loss = 0.0
        accuracy = 0.0

        model.train()

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            log_ps = model(images)

            ps = torch.exp(log_ps)
            _, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor))

            loss = criterion(log_ps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        train_accuracy = accuracy / len(train_loader)

        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        print(
            f"Epoch {epoch+1}/{epochs}.. "
            f"Train loss: {train_loss:.3f}.. "
            f"Train accuracy: {train_accuracy:.3f}.. "
        )

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss over Epochs")

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label="Train Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Accuracy over Epochs")

    plt.show()

    model.eval()

    x_test = test.drop(columns=["label"])
    x_test = x_test.values.reshape(-1, 1, IMAGE_SIZE, IMAGE_SIZE).astype(np.float32)
    x_test = torch.from_numpy(x_test).to(device)

    with torch.no_grad():
        ps = model(x_test)
        prediction = torch.argmax(ps, 1)

        print(f"Predictions: {prediction}")


if __name__ == "__main__":
    main()
