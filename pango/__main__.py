import cv2

from pango.puzzle_finder import PuzzleFinder

if __name__ == "__main__":
    image = cv2.imread("data/sample.jpg")

    if image is None:
        raise ValueError("Image not found or unable to load.")

    puzzle_finder = PuzzleFinder(image)
    puzzle_image = puzzle_finder.find()

    cv2.imshow("Puzzle Image", puzzle_image)
    cv2.waitKey(0)
