import cv2
from cv2.typing import MatLike

THRESHOLD_BLOCK_SIZE = 11
THRESHOLD_C = 2


class PuzzleFinder:
    def __init__(self, image: MatLike):
        self.image = image

    def find(self) -> MatLike:
        output = self._enhance_image(self.image.copy())

        return output

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

        return threshold_image
