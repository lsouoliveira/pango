from typing import Optional
import cv2
from cv2.typing import MatLike
from imutils.perspective import four_point_transform
from skimage.segmentation import clear_border
from dataclasses import dataclass

THRESHOLD_BLOCK_SIZE = 11
THRESHOLD_C = 2
MINIMUM_CONTOUR_AREA = 1000


class Error(Exception):
    pass


class NoPuzzleFoundError(Error):
    def __init__(self):
        super().__init__(f"No puzzle found.")


def draw_contours(image: MatLike, contours: list[MatLike]) -> MatLike:
    for contour in contours:
        cv2.drawContours(image, [contour], -1, (0, 255, 0), 3)

    return image


@dataclass
class ExtractedPuzzleImageResult:
    def __init__(self, image: MatLike, enhanced: MatLike, contour: MatLike):
        self.image = image
        self.enhanced = enhanced
        self.contour = contour


class PuzzleImageFinder:
    def __init__(self, image: MatLike):
        self.image = image

    def find(self) -> ExtractedPuzzleImageResult:
        output = self._enhance_image(self.image.copy())
        contours = self._find_contours(output)

        puzzle_contour = self._find_puzzle_contour(contours)

        if puzzle_contour is None:
            raise NoPuzzleFoundError()

        return ExtractedPuzzleImageResult(
            image=self._cut_puzzle(self.image, puzzle_contour),
            enhanced=self._cut_puzzle(output, puzzle_contour),
            contour=puzzle_contour,
        )

    def _enhance_image(self, image: MatLike) -> MatLike:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred_image = cv2.GaussianBlur(gray_image, (7, 7), 3)
        threshold_image = cv2.adaptiveThreshold(
            blurred_image,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            THRESHOLD_BLOCK_SIZE,
            THRESHOLD_C,
        )
        threshold_image = cv2.bitwise_not(threshold_image)

        return threshold_image

    def _find_contours(self, image: MatLike) -> list[MatLike]:
        contours, _ = cv2.findContours(
            image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        contours = [c for c in contours if cv2.contourArea(c) > MINIMUM_CONTOUR_AREA]

        return contours

    def _find_puzzle_contour(self, contours: list[MatLike]) -> Optional[MatLike]:
        for contour in contours:
            perimeter = cv2.arcLength(contour, True)
            approximation = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

            if perimeter < 100:
                continue

            if len(approximation) == 4:
                return approximation

    def _cut_puzzle(self, image: MatLike, contour: MatLike) -> MatLike:
        return four_point_transform(image, contour.reshape(4, 2))
