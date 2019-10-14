import os
import sys
import argparse

from src.labeling_tools.labeler import Labeler

def main(input_file, dictionary_file, output_file, to_lower=False):
    labeler = Labeler(dictionary_file, input_file, to_lower)
    labeler.match()
    labeler.write(output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file")
    parser.add_argument("dictionary_file")
    parser.add_argument("output_file")
    parser.add_argument("--to_lower", action='store_true')
    args = parser.parse_args()
    main(args.input_file, args.dictionary_file, args.output_file, args.to_lower)
