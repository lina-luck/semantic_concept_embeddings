import argparse
import sys
import os

project_path = os.path.split(os.path.abspath(os.path.realpath(__file__)))[0] + "/../"
sys.path.append(os.path.abspath(project_path))
from src.finetune_utils import *
from transformers import AutoModelForMaskedLM


def run(args):
    logging.info("load words")
    words = []
    with open(args.w_file, "r", encoding="utf-8") as f:
        for line in f:
            words.append(line.strip().lower())

    if args.bert_version.endswith('/'):
        args.bert_version = args.bert_version[:-1]

    logging.info("total number of words: " + str(len(words)))
    use_gpu = torch.cuda.is_available()

    bert_tokenizer = AutoTokenizer.from_pretrained(args.bert_version)
    model = AutoModelForMaskedLM.from_pretrained(args.bert_version, output_hidden_states=True)

    if not os.path.exists(os.path.join(args.out_path, os.path.basename(args.bert_version))):
        os.makedirs(os.path.join(args.out_path, os.path.basename(args.bert_version)))

    logging.info("extracting...")
    model.eval()
    if use_gpu:
        model.to("cuda")
    cnt = 0
    for w in words:
        if cnt % 500 == 0:
            logging.info(str(cnt) + "/" + str(len(words)) + " processed.")

        # load sentences
        w_sent = []
        with open(os.path.join(args.sent_path, w + ".txt"), "r", encoding="utf-8") as f:
            for line in f:
                sent = line.strip().lower()
                w_sent.append(sent)

        # build dataloader
        sent_i = torch.arange(len(w_sent))
        loader = DataLoader(dataset=TensorDataset(sent_i), batch_size=args.batch_size)

        emb_wi = torch.tensor([])
        for batch in loader:
            wi_batch = batch[0]
            if wi_batch.size()[0] == 1:
                sentences_batch = [np.array(w_sent)[wi_batch]]
            else:
                sentences_batch = np.array(w_sent)[wi_batch]

            words_batch = [w] * len(sentences_batch)

            input_tuple = tokenize_mask(sentences_batch, words_batch, args.max_seq_len, bert_tokenizer)
            token_ids, input_masks, indices = convert_tuple_to_tensors(input_tuple, use_gpu)
            with torch.no_grad():
                emb = model(input_ids=token_ids, attention_mask=input_masks).hidden_states[-1].detach().cpu()
            sent_ids = torch.tensor(list(range(emb.shape[0])), dtype=torch.long)
            hidden_mask = emb[sent_ids, indices]
            emb_wi = torch.cat((emb_wi, hidden_mask))
            del emb
            del token_ids, input_masks, indices
            torch.cuda.empty_cache()
        torch.save(emb_wi, os.path.join(args.out_path, os.path.basename(args.bert_version), w + ".pt"))
        cnt += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parameters about model
    parser.add_argument("-bert_version", help="bert version", default="bert-base-uncased")

    # parameters about IO
    parser.add_argument("-out_path", help="output path", default="mv")
    parser.add_argument("-w_file", help="words file", default="../words.txt")
    parser.add_argument("-sent_path", help="sentences path", default="../sents")

    # parameters about extracting
    parser.add_argument("-batch_size", help="batch size", default=6, type=int)
    parser.add_argument("-max_seq_len", help="max sequence length", default=16, type=int)

    args = parser.parse_args()
    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path)

    log_file_path = init_logging_path("log", "extract_mv")
    logging.basicConfig(filename=log_file_path,
                        level=10,
                        format='%(asctime)s %(levelname)s %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p')

    run(args)

    logging.info("done")
