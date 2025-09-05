from cv2.typing import MatLike
from enum import Enum
import cv2


class ConnectionSymbol(Enum):
    EQUAL = 1
    DIFFERENT = 2
    BLANK = 3

    def __str__(self):
        return self.name.lower()


class ConnectionClassifier:
    def __init__(self, image: MatLike):
        self.image = image

    def classify(self) -> ConnectionSymbol:
        if self.is_equal(self.image):
            return ConnectionSymbol.EQUAL
        elif self.is_different():
            return ConnectionSymbol.DIFFERENT
        else:
            return ConnectionSymbol.BLANK

    def is_equal(self, image: MatLike) -> bool:
        contours, _ = cv2.findContours(
            image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        rectangles = [
            cv2.boundingRect(contour)
            for contour in contours
            if cv2.contourArea(contour) > 50
        ]

        return len(rectangles) == 2

    def is_different(self) -> bool:
        contours, _ = cv2.findContours(
            self.image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        for contour in contours:
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
            area = cv2.contourArea(contour)

            if len(approx) > 5 and area > 100:
                return True

        return False
