import cv2
from cv2.typing import MatLike
from skimage.segmentation import clear_border


class ImageNormalizer:
    def __init__(
        self,
        input: MatLike,
        output_size: tuple[int, int] = (64, 64),
        object_size: tuple[int, int] = (48, 48),
    ):
        self.input = input
        self.output_size = output_size
        self.object_size = object_size

    def normalize(self) -> MatLike:
        output = self.convert_to_black_and_white(self.input)
        output = self.trim(output)

        return self.resize(output)

    def convert_to_black_and_white(self, input: MatLike) -> MatLike:
        output = cv2.cvtColor(input, cv2.COLOR_BGR2GRAY)
        output = cv2.adaptiveThreshold(
            output,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,
            2,
        )

        return cv2.bitwise_not(output)

    def trim(self, input: MatLike) -> MatLike:
        non_zero_coords = cv2.findNonZero(input)

        x, y, w, h = cv2.boundingRect(non_zero_coords)

        if w == 0 or h == 0:
            return input

        return input[y : y + h, x : x + w]

    def resize(self, input: MatLike) -> MatLike:
        scale = min(
            self.object_size[0] / input.shape[1],
            self.object_size[1] / input.shape[0],
        )

        new_width = int(input.shape[1] * scale)
        new_height = int(input.shape[0] * scale)

        resized = cv2.resize(input, (new_width, new_height))

        return cv2.copyMakeBorder(
            resized,
            top=(self.output_size[1] - new_height) // 2,
            bottom=(self.output_size[1] - new_height + 1) // 2,
            left=(self.output_size[0] - new_width) // 2,
            right=(self.output_size[0] - new_width + 1) // 2,
            borderType=cv2.BORDER_CONSTANT,
            value=[0, 0, 0],
        )
