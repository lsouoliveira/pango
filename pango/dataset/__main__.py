import argparse

import cv2

from pango.dataset.interactive_dataset_classifier import InteractiveDatasetClassifier
from pango.image_processing.cell_images_extractor import CellImagesExtractor
from pango.image_processing.connection_images_extractor import ConnectionImagesExtractor
from pango.image_processing.puzzle_image_finder import PuzzleImageFinder


def parse_args():
    parser = argparse.ArgumentParser(
        description="Classify images in a dataset interactively."
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory to save classified images.",
    )

    parser.add_argument(
        "--classes",
        type=str,
        nargs="+",
        required=True,
        help="List of class names for classification.",
    )

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the input image file.",
    )

    parser.add_argument(
        "--connections",
        action="store_true",
        help="Whether to classify connection images instead of cell images.",
    )

    return parser.parse_args()


def load_image(file_path: str):
    image = cv2.imread(file_path)

    if image is None:
        raise ValueError(f"Could not load image from path: {file_path}")

    return image


def main():
    args = parse_args()
    image = load_image(args.input)

    puzzle_image = PuzzleImageFinder(image).find().image

    if args.connections:
        connections = ConnectionImagesExtractor(puzzle_image).extract()
        images = connections[0] + connections[1]
    else:
        images = CellImagesExtractor(puzzle_image).extract()

    classifier = InteractiveDatasetClassifier(images, args.output, args.classes)
    classifier.classify()


if __name__ == "__main__":
    main()
