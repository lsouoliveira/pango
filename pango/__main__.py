from pango.puzzle_image_solver import PuzzleImageSolverPipeline

if __name__ == "__main__":
    solver = PuzzleImageSolverPipeline.load_image("data/train.png")
    _, puzzle = solver.run()

    print(puzzle.to_json())
