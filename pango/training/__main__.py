import pandas as pd
import matplotlib.pyplot as plt
import random
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

# https://www.kaggle.com/code/jedrzejdudzicz/mnist-dataset-100-accuracy
# https://www.kaggle.com/code/enwei26/mnist-digits-pytorch-cnn-99

IMAGE_SIZE = 64


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


def normalize_data(X, Y):
    X /= 255

    Y = preprocessing.LabelEncoder().fit_transform(Y)
    Y = preprocessing.OneHotEncoder(sparse_output=False).fit_transform(Y.reshape(-1, 1))

    num_classes = Y.shape[1]

    print(f"Number of classes: {num_classes}")

    return X, Y


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

    train, test = train_test_split(data, test_size=0.2, random_state=42)

    print(f"Train shape: {train.shape}")
    print(f"Test shape: {test.shape}")

    show_data_analysis(train)

    Y_train = data["label"]
    X_train = data.drop(columns=["label"])

    show_data_analysis(pd.concat([X_train, Y_train], axis=1))

    X_train, Y_train = apply_data_augmentation(X_train, Y_train)
    X_train, Y_train = normalize_data(X_train, Y_train)


if __name__ == "__main__":
    main()
