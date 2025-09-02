from cv2.typing import MatLike
from skimage.segmentation import clear_border


class CellImagesExtractor:
    def __init__(self, input: MatLike):
        self.input = input

    def extract(self) -> list[MatLike]:
        cells = []

        cell_width = self.input.shape[1] // 6
        cell_height = self.input.shape[0] // 6

        for i in range(6):
            for j in range(6):
                x, y, w, h = j * cell_width, i * cell_height, cell_width, cell_height

                cell = self.input[y : y + h, x : x + w]
                cell = clear_border(cell)

                cells.append(cell)

        return cells
