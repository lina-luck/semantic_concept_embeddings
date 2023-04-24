import os
import argparse
import sys
project_path = os.path.split(os.path.abspath(os.path.realpath(__file__)))[0] + "/../"
sys.path.append(os.path.abspath(project_path))
from src.global_top_words_utils import *
import logging
from src.utils import init_logging_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # constant
    parser.add_argument("-k", help="ranking number", default=5, type=int)

    # mention vector
    parser.add_argument("-mv_path", help="mention vector path from BERT", default="../data/baseline_embedding/numberbatch-en.txt")
    parser.add_argument("-w_file", help="file name of words, not necessary", default="../data/ap/ap_words.txt")
    parser.add_argument("-use_cosine", help="if True, use cosine similarity to find top words; if False, use L2 distance",
                        action="store_true", default=True)
    parser.add_argument("-top_word_path", help="path to store top words", default="../data/ap/")
    parser.add_argument("-num_closest", help="closest k vectors are averaged when compute distance for top words",
                        default=1, type=int)

    args = parser.parse_args()

    log_file_path = init_logging_path("log", "topwords")
    logging.basicConfig(filename=log_file_path,
                        level=10,
                        format='%(asctime)s %(levelname)s %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p')
    logging.info(str(args))
    args.words = None
    if args.w_file is not None:
        args.words = []
        with open(args.w_file, 'r', encoding="utf-8") as f:
            for line in f:
                args.words.append(line.strip().lower())

    global_top_words(args.top_word_path, args.mv_path, args.k, args.use_cosine, words=args.words,
                     closest_k=args.num_closest)
