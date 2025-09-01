import cv2
from cv2.typing import MatLike

from pango.puzzle_image_finder import ExtractedPuzzleImageResult, PuzzleImageFinder


class PuzzleImageSolverPipeline:
    def __init__(self, image: MatLike):
        self.image = image
        self.output_image: MatLike | None = None

    def run(self):
        result = self.extract_puzzle_image(self.image)

        self.output_image = result.image

    def extract_puzzle_image(self, image: MatLike) -> ExtractedPuzzleImageResult:
        puzzle_finder = PuzzleImageFinder(image)

        return puzzle_finder.find()

    @staticmethod
    def load_image(image_path: str) -> "PuzzleImageSolverPipeline":
        image = cv2.imread(image_path)

        if image is None:
            raise ValueError("Image not found or unable to load.")

        return PuzzleImageSolverPipeline(image)
