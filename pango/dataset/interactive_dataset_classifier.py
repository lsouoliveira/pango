import os
import cv2
from cv2.typing import MatLike
import hashlib


class InteractiveDatasetClassifier:
    def __init__(
        self,
        input: list[MatLike],
        output_dir: str,
        classes: list[str],
    ):
        self.input = input
        self.output_dir = output_dir
        self.classes = classes

    def classify(self):
        self.ensure_output_dirs_exist()
        self.start_classification(self.input)

    def start_classification(self, images: list[MatLike]):
        curr = 0

        while curr < len(images):
            image = images[curr]

            key = self.show_image(image, len(images), curr)
            curr = self.handle_user_input(key, image, curr)

    def show_image(self, image: MatLike, total: int, index: int) -> int:
        cv2.imshow(
            f"Image {index + 1}/{total} - Press 1-{len(self.classes)} to classify, n to skip, Esc to exit",
            image,
        )

        key = cv2.waitKey(0)

        cv2.destroyAllWindows()

        return key

    def handle_user_input(self, key: int, image: MatLike, cursor: int = 0) -> int:
        if key == 27:
            cv2.destroyAllWindows()
            exit(0)

        if key == ord("n"):
            return cursor + 1

        if key == 8:
            return max(0, cursor - 1)

        for i, class_name in enumerate(self.classes):
            if key == ord(str(i + 1)):
                self.save_image(image, class_name)

                return cursor + 1

        raise ValueError(f"Invalid key: {key}")

    def save_image(self, image: MatLike, class_name: str):
        filename = hashlib.sha256(image.tobytes()).hexdigest()
        path = f"{self.output_dir}/{class_name}/{filename}.png"

        cv2.imwrite(path, image)

        print(f"Saved image to {filename}, size: {image.shape}, class: {class_name}")

    def ensure_output_dirs_exist(self):
        for class_name in self.classes:
            dir_path = f"{self.output_dir}/{class_name}"

            os.makedirs(dir_path, exist_ok=True)
