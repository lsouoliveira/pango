from cv2.typing import MatLike
from skimage.segmentation import clear_border

from pango.image_processing.image_normalizer import ImageNormalizer

CONNECTION_WIDTH = 40
CONNECTION_PADDING_RATIO = 0.25


class ConnectionImagesExtractor:
    def __init__(self, input: MatLike):
        self.input = input

    def extract(self) -> tuple[list[MatLike], list[MatLike]]:
        return (
            self._extract_vertical_connections(),
            self._extract_horizontal_connections(),
        )

    def _extract_vertical_connections(self) -> list[MatLike]:
        connections = []

        for i in range(6):
            for j in range(5):
                x = (j + 1) * (self.input.shape[1] // 6) - CONNECTION_WIDTH // 2
                y = i * (self.input.shape[0] // 6)
                w = CONNECTION_WIDTH
                h = self.input.shape[0] // 6

                padding = int(h * CONNECTION_PADDING_RATIO)
                connection = self.input[y + padding : y + h - padding, x : x + w]

                connections.append(connection)

        connections = [ImageNormalizer(conn).normalize() for conn in connections]

        return connections

    def _extract_horizontal_connections(self) -> list[MatLike]:
        connections = []

        for i in range(5):
            for j in range(6):
                x = j * (self.input.shape[1] // 6)
                y = (i + 1) * (self.input.shape[0] // 6) - CONNECTION_WIDTH // 2
                w = self.input.shape[1] // 6
                h = CONNECTION_WIDTH

                padding = int(w * CONNECTION_PADDING_RATIO)
                connection = self.input[y : y + h, x + padding : x + w - padding]

                connections.append(connection)

        connections = [ImageNormalizer(conn).normalize() for conn in connections]

        return connections
