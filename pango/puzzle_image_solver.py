import cv2
from cv2.typing import MatLike

from pango.cell_images_extractor import CellImagesExtractor
from pango.puzzle_image_finder import ExtractedPuzzleImageResult, PuzzleImageFinder


class PuzzleImageSolverPipeline:
    def __init__(self, image: MatLike):
        self.image = image
        self.output_image: MatLike | None = None

    def run(self):
        result = self.extract_puzzle_image(self.image)
        cell_images = self.extract_cell_images(result.enhanced)

        self.output_image = result.image

    def extract_puzzle_image(self, image: MatLike) -> ExtractedPuzzleImageResult:
        puzzle_finder = PuzzleImageFinder(image)

        return puzzle_finder.find()

    def extract_cell_images(self, puzzle_image: MatLike) -> list[MatLike]:
        extractor = CellImagesExtractor(puzzle_image)

        return extractor.extract()

    @staticmethod
    def load_image(image_path: str) -> "PuzzleImageSolverPipeline":
        image = cv2.imread(image_path)

        if image is None:
            raise ValueError("Image not found or unable to load.")

        return PuzzleImageSolverPipeline(image)
