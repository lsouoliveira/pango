import cv2

from pango.puzzle_image_solver import PuzzleImageSolverPipeline

if __name__ == "__main__":
    solver = PuzzleImageSolverPipeline.load_image("data/sample.jpg")
    solver.run()

    if solver.output_image is None:
        raise ValueError("No output image available.")

    cv2.imshow("Puzzle Image", solver.output_image)
    cv2.waitKey(0)
