import os
import cv2
import glob
import uuid

from cv2.typing import MatLike

from pango.cell_images_extractor import CellImagesExtractor
from pango.puzzle_image_finder import PuzzleImageFinder

IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png"]


class InteractiveDatasetBuilder:
    def __init__(
        self,
        classes: list[str],
        input_dir: str,
        output_dir: str,
        image_size: tuple[int, int] = (300, 300),
    ):
        self.classes = classes
        self.output_dir = output_dir
        self.image_size = image_size
        self.input_dir = input_dir

    def build(self):
        self.create_output_dirs()
        input_images = self.load_input_images()

        for image in input_images:
            puzzle = PuzzleImageFinder(image)
            puzzle_result = puzzle.find()

            cv2.imshow("Puzzle", puzzle_result.image)
            key = cv2.waitKey(0)

            if key == 27:
                continue

            cells = CellImagesExtractor(puzzle_result.enhanced).extract()

            self.classify_cells(cells)

    def list_input_file_paths(self) -> list[str]:
        files = []

        for ext in IMAGE_EXTENSIONS:
            files.extend(glob.glob(f"{self.input_dir}/**/*{ext}", recursive=True))

        return files

    def create_output_dirs(self):
        for label in self.classes:
            dir_path = os.path.join(self.output_dir, label)

            os.makedirs(dir_path, exist_ok=True)

    def load_input_images(self) -> list[MatLike]:
        return [
            self.load_image(image_path) for image_path in self.list_input_file_paths()
        ]

    def extract_cells(self, image: MatLike) -> list[MatLike]:
        extractor = CellImagesExtractor(image)

        return extractor.extract()

    def classify_cells(self, cells: list[MatLike]):
        for cell in cells:
            label = self.classify_cell(cell)

            self.save_cell_image(cell, label)

    def classify_cell(self, cell: MatLike) -> str:
        while True:
            cv2.imshow("Cell", cell)

            key = cv2.waitKey(0)

            index = key - ord("1")

            if 0 <= index < len(self.classes):
                return self.classes[index]

    def save_cell_image(self, cell: MatLike, label: str):
        cell_resized = cv2.resize(cell, self.image_size)

        filename = f"{uuid.uuid4()}.png"
        filepath = os.path.join(self.output_dir, label, filename)

        cv2.imwrite(filepath, cell_resized)

        print(f"Saved cell image to {filepath}")

    def load_image(self, image_path: str) -> MatLike:
        image = cv2.imread(image_path)

        if image is None:
            raise ValueError(f"Image not found or unable to load: {image_path}")

        return image
