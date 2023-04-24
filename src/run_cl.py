import argparse
import sys
import os
project_path = os.path.split(os.path.abspath(os.path.realpath(__file__)))[0] + "/../"
sys.path.append(os.path.abspath(project_path))

from src.cl_utils import *
from src.cnn_utils import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("-w_file", help="file name of words", default="../data/battig/battig_words.txt")
    parser.add_argument("-mv_path", help="mention vector path from BERT", default="mv_mirrorbert/battig_all.pt")
                        # default="/home_stockage/cril/bouraoui_group/LiNa/mention_vectors/ijcai2021_mv_bert_large/bert_mask_mv/bert_mask_last/")

    # parameters for building positive pairs
    parser.add_argument("-mv_knn_file", help="knn file in mention vector space", default="mv_knn/battig/cos_20.npy")
    parser.add_argument("-word_knn_file", help="file name of knn words of mention vectors",
                        default="../data/test/closest5_top_words_cos.json.bz2")

    # parameters for model training
    parser.add_argument("-in_dim", help="input dimension", default=768, type=int)

    # other parameters
    parser.add_argument("-out_path", help="output path to store facet embedding", default="cl_out")

    # parameter for dataset
    parser.add_argument("-dataset", help="dataset", default="battig")

    parser.add_argument("-nonlinear", help='whether to use non-linear transformation if MLP', default=False,
                        action='store_true')

    parser.add_argument("-method", help='method to build positive mv pairs', default=1, choices=[1, 2, 3], type=int)
    parser.add_argument("-batch_size", help='batch size for bert model', default=32, type=int)
    parser.add_argument("-k", help='number of neighbours', default=10, type=int)
    parser.add_argument("-lr", help='learning rate for bert model', default=2e-3, type=float)

    parser.add_argument("-weight_decay", help='weight decay', default=1e-3, type=float)
    parser.add_argument("-use_hard_pair", help='whether use hard pair when computing contrastive loss', default=True,
                        action='store_true')
    parser.add_argument("-tau", help='hyper-parameter of contrastive loss', default=0.05, type=float)
    parser.add_argument("-lr_schedule", help="linear or cosine learning rate schedule", default='cosine',
                        choices=['linear', 'cosine'])
    parser.add_argument('-hidden_dim', help='hidden dimension of MLP', default=256, type=int)
    parser.add_argument('-out_dim', help='output dimension of MLP', default=64, type=int)
    parser.add_argument("-olp_th", help='overlap threshold', default=0.5, type=float)
    parser.add_argument("-use_cosine",
                        help='whether use cosine to compute similarity between mv, if False, use L2 distance',
                        default=True, action='store_true')

    args = parser.parse_args()

    if not os.path.exists(os.path.join(args.out_path, args.dataset)):
        os.makedirs(os.path.join(args.out_path, args.dataset))

    log_file_path = init_logging_path("log", "contrastive_learning")
    logging.basicConfig(filename=log_file_path,
                        level=10,
                        format='%(asctime)s %(levelname)s %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p')
    word_list = run(args)[1]

    with open(os.path.join("../data", args.dataset, "word_list.txt"), "w", encoding="utf-8") as f:
        f.write('\n'.join(word_list))