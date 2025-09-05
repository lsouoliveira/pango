from cv2.typing import MatLike
from enum import Enum
import math
import cv2


class Shape(Enum):
    SUN = 0
    MOON = 1
    UNKNOWN = 2

    def __str__(self):
        return self.name.lower()


class ShapeClassifier:
    def __init__(self, image: MatLike):
        self.image = image

    def classify(self) -> Shape:
        if self._is_sun(self.image):
            return Shape.SUN
        elif self._is_moon(self.image):
            return Shape.MOON
        else:
            return Shape.UNKNOWN

    def _is_sun(self, image: MatLike) -> bool:
        contours = self._extract_contours(image)

        for contour in contours:
            area = cv2.contourArea(contour)

            if area == 0:
                continue

            perimeter = cv2.arcLength(contour, True)

            if perimeter == 0:
                continue

            circularity = 4 * math.pi * (area / (perimeter * perimeter))

            if circularity > 0.7:
                return True

        return False

    def _is_moon(self, image: MatLike) -> bool:
        contours = self._extract_contours(image)

        for contour in contours:
            area = cv2.contourArea(contour)

            if area == 0:
                continue

            perimeter = cv2.arcLength(contour, True)

            if perimeter == 0:
                continue

            approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)

            if len(approx) > 5 and area >= 400:
                return True

        return False

    def _extract_contours(self, image: MatLike):
        contours, _ = cv2.findContours(
            image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        return sorted(contours, key=cv2.contourArea, reverse=True)
