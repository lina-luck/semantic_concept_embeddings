import argparse
import sys
import os

project_path = os.path.split(os.path.abspath(os.path.realpath(__file__)))[0] + "/../"
sys.path.append(os.path.abspath(project_path))

from src.finetune_utils import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("-w_file", help="file name of words", default="../data/test/test.txt")
    parser.add_argument("-mv_path", help="mention vector path from BERT", default="../mv/bert-base-uncased")
                        
    parser.add_argument("-sent_path", help="sentences path", default="../sents")
    parser.add_argument("-bert_version", help="bert version or bert path",
                        default="../pretrained_models/bert-base-uncased")

    # parameters for building positive pairs
    parser.add_argument("-mv_knn_file", help="knn file in mention vector space", default="mv_knn/test/cos_20.npy")
    parser.add_argument("-word_knn_file", help="file name of knn words of mention vectors",
                        default="../data/test/closest5_top_words_cos.json.bz2")

    # other parameters
    parser.add_argument("-out_path", help="output path to store finetuned bert embedding", default="fine_out")

    # parameter for dataset
    parser.add_argument("-dataset", help="dataset", default="test")

    # parameters for finetuning model
    parser.add_argument("-method", help='method to build positive mv pairs', default=3, choices=[1, 2, 3], type=int)
    parser.add_argument("-k", help='number of neighbours', default=3, type=int)
    parser.add_argument("-use_cosine", help='whether use cosine to compute similarity between mv, if False, use L2 distance',
                        default=False, action='store_true')
    parser.add_argument("-olp_th", help='overlap threshold', default=0.1, type=float)
    parser.add_argument("-batch_size", help='batch size for bert model', default=32, type=int)
    parser.add_argument("-lr", help='learning rate for bert model', default=2e-5, type=float)
    parser.add_argument("-weight_decay", help='weight decay', default=1e-3, type=float)
    parser.add_argument("-use_hard_pair", help='whether use hard pair when computing contrastive loss', default=False, action='store_true')
    parser.add_argument("-tau", help='hyper-parameter of contrastive loss', default=0.05, type=float)
    parser.add_argument("-margin", help='hyper-parameter of contrastive loss', default=0.05, type=float)
    parser.add_argument('-use_trans', help='if True, add 2 linear mapping layers to the top of bert model', default=False, action='store_true')
    parser.add_argument("-lr_schedule", help="linear or cosine learning rate schedule", default='cosine',
                        choices=['linear', 'cosine'])
    parser.add_argument("-loss", help="loss function", default='infonce', choices=['infonce', 'triple'])
    parser.add_argument("-nonlinear", help='whether to use non-linear transformation if MLP', default=False, action='store_true')
    parser.add_argument('-hidden_dim', help='hidden dimension of MLP', default=256, type=int)
    parser.add_argument('-out_dim', help='output dimension of MLP', default=64, type=int)
    parser.add_argument('-max_seq_len', help='max sequence length', default=32, type=int)

    args = parser.parse_args()

    log_file_path = init_logging_path("log", "finetune_bert")
    logging.basicConfig(filename=log_file_path,
                        level=10,
                        format='%(asctime)s %(levelname)s %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p')

    result_path = os.path.join(os.path.abspath(project_path), 'result_finetune_bert', args.dataset, os.path.basename(args.bert_version))
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    finetune_model(args)
