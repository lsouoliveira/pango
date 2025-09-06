import cv2
import glob
import pandas as pd


ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}


class DatasetPacker:
    def __init__(
        self,
        input_dir: str,
        output_file: str,
        classes: list[str],
        image_size: tuple[int, int] = (64, 64),
    ):
        self.input_dir = input_dir
        self.output_file = output_file
        self.classes = classes
        self.image_size = image_size

    def pack(self):
        class_images = self._load_classes_images()

        self.export_classes_to_csv(class_images)

    def _load_classes_images(self):
        class_images = {}

        for class_name in self.classes:
            class_images[class_name] = self._load_images_by_class(class_name)

        return class_images

    def _load_images_by_class(self, class_name: str):
        image_filenames = self._get_class_image_filenames(class_name)

        return [self._load_image(filename) for filename in image_filenames]

    def _get_class_image_filenames(self, class_name: str):
        class_dir = f"{self.input_dir}/{class_name}"
        image_paths = []

        for ext in ALLOWED_EXTENSIONS:
            image_paths.extend(glob.glob(f"{class_dir}/*{ext}"))

        return image_paths

    def _load_image(self, image_path: str):
        image = cv2.imread(image_path)

        if image is None:
            raise ValueError(f"Could not read image: {image_path}")

        return image

    def export_classes_to_csv(self, class_images: dict):
        rows = []

        for class_name, images in class_images.items():
            for image in images:
                row = [class_name]

                resized_image = cv2.resize(image, self.image_size)
                first_channel = resized_image[:, :, 0]

                flattened = first_channel.flatten().tolist()

                row.extend(flattened)

                rows.append(row)

        column_names = ["label"] + [
            f"p{i}" for i in range(self.image_size[0] * self.image_size[1])
        ]

        df = pd.DataFrame(rows, columns=column_names)
        df.to_csv(self.output_file, index=False)


if __name__ == "__main__":
    packer = DatasetPacker(
        input_dir="dataset",
        output_file="dataset.csv",
        classes=["bh", "blank", "bv", "eh", "moon", "sun", "xh", "xv"],
    )
    packer.pack()
