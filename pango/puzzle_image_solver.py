from typing import Optional
import cv2
from cv2.typing import MatLike

from pango.image_processing.cell_images_extractor import CellImagesExtractor
from pango.image_processing.connection_classifier import (
    ConnectionSymbol,
    ConnectionClassifier,
)
from pango.image_processing.connection_images_extractor import ConnectionImagesExtractor
from pango.image_processing.puzzle_image_finder import (
    ExtractedPuzzleImageResult,
    PuzzleImageFinder,
)
from pango.image_processing.shape_classifier import Shape, ShapeClassifier
from pango.puzzle import (
    Cell,
    CellValue,
    Connection,
    ConnectionType,
    DifferentConnection,
    EqualConnection,
    Puzzle,
    PuzzleGrid,
    SymbolType,
)

SHAPE_MAPPING = {
    Shape.SUN: SymbolType.SUN,
    Shape.MOON: SymbolType.MOON,
}

CONNECTION_MAPPING = {
    ConnectionSymbol.EQUAL: ConnectionType.EQUAL,
    ConnectionSymbol.DIFFERENT: ConnectionType.DIFFERENT,
}


def map_shape_to_symbol(shape: Shape) -> CellValue:
    return SHAPE_MAPPING.get(shape, None)


def map_connection_to_connection_type(
    conn: ConnectionSymbol,
) -> Optional[ConnectionType]:
    return CONNECTION_MAPPING.get(conn, None)


def create_connection(
    src: Cell, dst: Cell, connection_type: ConnectionType
) -> Connection:
    if connection_type == ConnectionType.EQUAL:
        return EqualConnection(src, dst)
    elif connection_type == ConnectionType.DIFFERENT:
        return DifferentConnection(src, dst)


class Error(Exception):
    pass


class InvalidPuzzle(Error):
    def __init__(self):
        super().__init__("The provided puzzle is invalid.")


class InvalidNumberOfConnections(Error):
    def __init__(self, connections_count: int = 0):
        super().__init__(f"Invalid number of connections: {connections_count}.")


class PuzzleImageSolverPipeline:
    def __init__(self, image: MatLike):
        self.image = image

    def run(self):
        result = self.extract_puzzle_image(self.image)

        cell_images = self.extract_cell_images(result.image)
        connection_images = self._extract_connection_images(result.image)

        # puzzle = self.build_puzzle(shapes, connections)
        #
        # if not puzzle.is_valid():
        #     raise InvalidPuzzle()
        #
        # puzzle.solve()
        #
        # return (result.image, puzzle)

    def extract_puzzle_image(self, image: MatLike) -> ExtractedPuzzleImageResult:
        puzzle_finder = PuzzleImageFinder(image)

        return puzzle_finder.find()

    def extract_cell_images(self, puzzle_image: MatLike) -> list[MatLike]:
        extractor = CellImagesExtractor(puzzle_image)

        return extractor.extract()

    def _extract_connection_images(
        self, puzzle_image: MatLike
    ) -> tuple[list[MatLike], list[MatLike]]:
        extractor = ConnectionImagesExtractor(puzzle_image)

        return extractor.extract()

    def build_puzzle(
        self,
        shapes: list[Shape],
        connections: tuple[list[ConnectionSymbol], list[ConnectionSymbol]],
    ) -> Puzzle:
        grid: PuzzleGrid = [[Cell(i, j) for j in range(6)] for i in range(6)]
        vertical_connections, horizontal_connections = connections
        puzzle_connections = []

        for i in range(6):
            for j in range(6):
                shape = shapes[i * 6 + j]

                grid[i][j].value = map_shape_to_symbol(shape)

        for i in range(6):
            for j in range(5):
                shape = vertical_connections[i * 5 + j]
                connection = map_connection_to_connection_type(shape)

                if connection is None:
                    continue

                new_connection = create_connection(
                    grid[i][j], grid[i][j + 1], connection
                )

                puzzle_connections.append(new_connection)

        for i in range(5):
            for j in range(6):
                shape = horizontal_connections[i * 6 + j]
                connection = map_connection_to_connection_type(shape)

                if connection is None:
                    continue

                new_connection = create_connection(
                    grid[i][j], grid[i + 1][j], connection
                )

                puzzle_connections.append(new_connection)

        return Puzzle(grid, puzzle_connections)

    @staticmethod
    def load_image(image_path: str) -> "PuzzleImageSolverPipeline":
        image = cv2.imread(image_path)

        if image is None:
            raise ValueError("Image not found or unable to load.")

        return PuzzleImageSolverPipeline(image)
