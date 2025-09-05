from enum import Enum
import json


class SymbolType(Enum):
    SUN = 0
    MOON = 1


SYMBOLS = [SymbolType.SUN, SymbolType.MOON]


CellValue = SymbolType | None
PuzzleGrid = list[list["Cell"]]


class Error(Exception):
    pass


class NoSolutionFound(Error):
    def __init__(self):
        super().__init__("No solution found.")


class Cell:
    def __init__(
        self,
        row: int,
        col: int,
        value: CellValue = None,
    ):
        self.row = row
        self.col = col
        self.value = value

    def is_empty(self) -> bool:
        return self.value is None

    def is_filled(self) -> bool:
        return self.value is not None

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Cell):
            return False

        return self.value == other.value


class ConnectionType(Enum):
    EQUAL = 0
    DIFFERENT = 1


class Connection:
    def __init__(self, src: Cell, dst: Cell, connection_type: ConnectionType):
        self.src = src
        self.dst = dst
        self.connection_type = connection_type

    def is_valid(self) -> bool:
        raise NotImplementedError("is_valid method not implemented yet.")


class EqualConnection(Connection):
    def __init__(self, src: Cell, dst: Cell):
        super().__init__(src, dst, ConnectionType.EQUAL)

    def is_valid(self) -> bool:
        if self.src.value is None or self.dst.value is None:
            return True

        return self.src.value == self.dst.value


class DifferentConnection(Connection):
    def __init__(self, src: Cell, dst: Cell):
        super().__init__(src, dst, ConnectionType.DIFFERENT)

    def is_valid(self) -> bool:
        if self.src.value is None or self.dst.value is None:
            return True

        return self.src.value != self.dst.value


def next_empty_cell(cells: list[Cell]) -> Cell | None:
    for cell in cells:
        if cell.is_empty():
            return cell

    return None


class PuzzleValidator:
    def __init__(self, puzzle: "Puzzle"):
        self._puzzle = puzzle

    def validate(self) -> bool:
        for cells in self._all_ranges():
            if not self._validate_at_max_three_identical_symbols(cells):
                return False

            if not self.validate_no_more_than_two_identical_symbols_adjacent(cells):
                return False

        for connection in self._puzzle.connections:
            if not connection.is_valid():
                return False

        return True

    def validate_no_more_than_two_identical_symbols_adjacent(
        self, cells: list[Cell]
    ) -> bool:
        for i in range(2, len(cells)):
            if (
                cells[i].value is not None
                and cells[i] == cells[i - 1]
                and cells[i] == cells[i - 2]
            ):
                return False

        return True

    def _validate_at_max_three_identical_symbols(self, cells: list[Cell]) -> bool:
        counts = {SymbolType.SUN: 0, SymbolType.MOON: 0}

        for cell in cells:
            if cell.value is not None:
                counts[cell.value] += 1

                if counts[cell.value] > 3:
                    return False

        return True

    def _rows(self) -> list[list[Cell]]:
        return [self._puzzle[row] for row in range(len(self._puzzle._grid))]

    def _columns(self) -> list[list[Cell]]:
        return [
            [self._puzzle[row][col] for row in range(len(self._puzzle._grid))]
            for col in range(len(self._puzzle._grid[0]))
        ]

    def _all_ranges(self) -> list[list[Cell]]:
        return self._rows() + self._columns()

    def _get_row(self, row: int) -> list[Cell]:
        return self._puzzle[row]

    def _get_column(self, col: int) -> list[Cell]:
        return [self._puzzle[row][col] for row in range(len(self._puzzle._grid))]


class Puzzle:
    def __init__(self, grid: PuzzleGrid, connections: list[Connection] = []):
        self._grid = grid
        self._connections = connections

    def solve(self):
        if self.is_solved():
            return

        empty_cells = self.empty_cells()
        stack = []

        stack.append((next_empty_cell(empty_cells), [SymbolType.SUN, SymbolType.MOON]))

        while len(stack) > 0:
            cell, symbols = stack[-1]

            if len(symbols) == 0:
                cell.value = None
                stack.pop()
                continue

            symbol = symbols.pop()
            cell.value = symbol

            if self.is_valid():
                next_cell = next_empty_cell(empty_cells)

                if next_cell is None:
                    return

                stack.append((next_cell, [SymbolType.SUN, SymbolType.MOON]))

        raise NoSolutionFound()

    def is_valid(self) -> bool:
        return PuzzleValidator(self).validate()

    def is_solved(self) -> bool:
        for row in self._grid:
            for cell in row:
                if cell.is_empty():
                    return False

        return self.is_valid()

    def empty_cells(self) -> list[Cell]:
        return [cell for row in self._grid for cell in row if cell.is_empty()]

    @property
    def connections(self) -> list[Connection]:
        return self._connections

    def __getitem__(self, index: int) -> list[Cell]:
        return self._grid[index]

    def __repr__(self):
        return str([[cell.value for cell in row] for row in self._grid])

    def to_json(self) -> str:
        grid_representation = [
            [cell.value.name if cell.value else None for cell in row]
            for row in self._grid
        ]
        connections_representation = [
            {
                "src": {"row": conn.src.row, "col": conn.src.col},
                "dst": {"row": conn.dst.row, "col": conn.dst.col},
                "type": conn.connection_type.name,
            }
            for conn in self._connections
        ]

        puzzle_dict = {
            "grid": grid_representation,
            "connections": connections_representation,
        }

        return json.dumps(puzzle_dict, indent=4)
