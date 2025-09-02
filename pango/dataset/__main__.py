import argparse

from pango.dataset.interactive_dataset_builder import InteractiveDatasetBuilder


def parse_args():
    parser = argparse.ArgumentParser(description="A interactive dataset builder")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input directory containing images to classify",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory to save classified images",
    )
    parser.add_argument(
        "--classes",
        type=str,
        nargs="+",
        required=True,
        help="List of class names for classification",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    builder = InteractiveDatasetBuilder(
        input_dir=args.input, output_dir=args.output, classes=args.classes
    )

    builder.build()


if __name__ == "__main__":
    main()
