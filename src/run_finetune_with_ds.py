import os
import sys
import argparse
project_path = os.path.split(os.path.abspath(os.path.realpath(__file__)))[0] + "/../"
sys.path.append(os.path.abspath(project_path))

from src.finetune_with_ds_utils import *
from src.utils import init_logging_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("-prop_file", help="file name of properties", default="../data/ds_data/properties.txt")
    parser.add_argument("-prop_num", help="properties number expected to use", default=5, type=int)

    parser.add_argument("-ppt_path", help="path of sentences of properties", default="../data/ds_data/prop_sentences")

    parser.add_argument("-bert_version", help="bert version or bert path", default="bert-base-uncased")

    parser.add_argument("-batch_size", help='batch size for bert model', default=2, type=int)
    parser.add_argument("-lr", help='learning rate for bert model', default=2e-4, type=float)
    parser.add_argument("-weight_decay", help='weight decay', default=1e-3, type=float)
    parser.add_argument("-use_hard_pair", help='whether use hard pair when computing contrastive loss', default=False,
                        action='store_true')
    parser.add_argument("-tau", help='hyper-parameter of infonce loss', default=0.05, type=float)
    parser.add_argument("-margin", help='hyper-parameter of triple loss', default=0.05, type=float)
    parser.add_argument('-use_trans', help='if True, add 2 linear mapping layers to the top of bert model',
                        default=False, action='store_true')
    parser.add_argument('-hidden_dim', help='hidden dimension of MLP', default=512, type=int)
    parser.add_argument('-out_dim', help='output dimension of MLP', default=256, type=int)

    parser.add_argument("-loss", help='loss function', default='infonce', choices=['infonce', 'triple'])
    parser.add_argument("-max_seq_len", help='max sequence length', default=32, type=int)
    parser.add_argument("-lr_schedule", help="linear or cosine learning rate schedule", default='cosine', choices=['linear', 'cosine'])
    parser.add_argument("-nonlinear", help='whether to use non-linear transformation if MLP', default=False,
                        action='store_true')

    args = parser.parse_args()

    log_file_path = init_logging_path("log", "finetune_ds")
    logging.basicConfig(filename=log_file_path,
                        level=10,
                        format='%(asctime)s %(levelname)s %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p')


    finetune_with_ds(args)