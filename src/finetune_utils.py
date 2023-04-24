# util file for finetuning model
import sys
import os
project_path = os.path.split(os.path.abspath(os.path.realpath(__file__)))[0] + "/../"
sys.path.append(os.path.abspath(project_path))
from src.utils import *
from transformers import AutoTokenizer
from src.models import BertMaskModel, BertMaskModelWithMLP
import gc


def tokenize_mask_single(sent, word, max_seq_len, tokenizer):
    """
    Tokenize single sentence for BERT Masked Model or Roberta Masked model
    :param sent: a sentence
    :param word: target word to be masked
    :param max_seq_len:
    :param tokenizer:
    :return:
    list of token ids, input mask and indices of mention entity
    """
    cls_token = tokenizer.convert_ids_to_tokens(tokenizer.cls_token_id)  # "[CLS]"
    sep_token = tokenizer.convert_ids_to_tokens(tokenizer.sep_token_id)  # "[SEP]"
    mask_token = tokenizer.convert_ids_to_tokens(tokenizer.mask_token_id)  #"[MASK]"
    left_seq, _, right_seq = sent.partition(str(word))
    tokens = tokenizer.tokenize(left_seq)
    # if left part of word is longer than max_seq_len, remove first half of tokens
    if len(tokens) >= max_seq_len - 3:  # 3 -> cls, mask, sep
        tokens = tokens[-((max_seq_len-3) // 2):]
    tokens = [cls_token] + tokens
    idx = len(tokens)
    assert idx < max_seq_len
    tokens += [mask_token]
    tokens += tokenizer.tokenize(right_seq)
    if len(tokens) > max_seq_len - 1:  # 1 -> sep
        tokens = tokens[:max_seq_len - 1] + [sep_token]
    t_id = tokenizer.convert_tokens_to_ids(tokens)
    padding = [0] * (max_seq_len - len(t_id))
    i_mask = [1] * len(t_id) + padding
    t_id += padding
    return t_id, i_mask, idx


def tokenize_mask(sentences, words, max_seq_len, tokenizer):
    """
    Tokenize sentences for BERT Masked Model or Roberta Masked model
    :param sentences: sentence list
    :param words: target word list, corresponding to sentences
    :param max_seq_len:
    :param tokenizer:
    :return:
    """
    token_ids = []
    input_mask = []
    indices = []
    for sent, word in zip(sentences, words):
        tid, im, idx = tokenize_mask_single(sent, word, max_seq_len, tokenizer)
        token_ids.append(tid)
        input_mask.append(im)
        indices.append(idx)
    return token_ids, input_mask, indices


def convert_tuple_to_tensors(input_tuple, use_gpu=False):
    """
    convert list in tuple (token_ids, input_masks, indices) to tensor separately
    :param input_tuple:
    :param use_gpu:
    :return:
    """
    token_ids, input_masks, indices = input_tuple

    token_ids = torch.tensor(token_ids, dtype=torch.long)
    input_masks = torch.tensor(input_masks, dtype=torch.long)
    indices = torch.tensor(indices, dtype=torch.long)

    if use_gpu:
        token_ids = token_ids.to("cuda")
        input_masks = input_masks.to("cuda")
        indices = indices.to("cuda")

    return token_ids, input_masks, indices


# main function for fine-tuning bert
def finetune_model(args):
    model_path = os.path.join('finetuned_bert_model', args.dataset)
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    pos_pair_path = os.path.join('pos_pair', args.dataset)
    if not os.path.exists(pos_pair_path):
        os.makedirs(pos_pair_path)

    method_dict = {1: "mv_knn", 2: "word_knn_js", 3: "word_knn_pi"}
    # loss
    if args.bert_version.endswith('/'):
        args.bert_version = args.bert_version[:-1]

    model_params = [method_dict[args.method], os.path.basename(args.bert_version), args.loss]
    if args.loss == "infonce":
        model_params.append(str(args.tau))
    elif args.loss == "triple":
        model_params.append(str(args.margin))
    # way to compute similarity relation between vectors
    if args.method == 1:
        model_params.append(str(args.k))
        if args.use_cosine:
            model_params.append("cos")
        else:
            model_params.append("L2")
    elif args.method in [2, 3]:
        model_params.extend([str(args.k), str(args.olp_th)])

    model_params += [str(args.lr), str(args.batch_size), str(args.weight_decay)]

    logging.info(str(args))

    # 1. load words
    if args.w_file is not None:
        words = []
        with open(args.w_file, 'r', encoding="utf-8") as f:
            for line in f:
                words.append(line.strip().lower())
    else:
        words = [f.split('.')[0] for f in os.listdir(args.mv_path)]

    # 2. load mention vectors
    logging.info("load in mention vectors")
    men_vec, word_list = load_vectors(args.mv_path, words)

    # 3. positive pairs of mv index
    logging.info("build data loader")
    mv_idx_pairs = build_positive_pair(men_vec, word_list, method=args.method, k=args.k,
                                       mv_knn_file=args.mv_knn_file,
                                       use_cosine=args.use_cosine, word_knn_file=args.word_knn_file,
                                       olp_th=args.olp_th, out_path=pos_pair_path)

    train_pairs, dev_pairs = train_test_split(mv_idx_pairs, train_size=0.8)
    train_labels = None
    if args.method == 1:
        train_labels = np.array(train_pairs)[:, 0]
    train_loader = dataloader(train_pairs, args.batch_size, is_train=True, labels=train_labels)
    dev_loader = dataloader(dev_pairs, args.batch_size)
    del men_vec
    del mv_idx_pairs
    del train_pairs
    del dev_pairs

    bert_tokenizer = AutoTokenizer.from_pretrained(args.bert_version)

    # load sentences
    logging.info("load sentences")
    sentences = load_sentences(words, args.sent_path, word_list)
    logging.info("total number of sentences: " + str(len(sentences)))

    # build model
    if args.use_trans:  # use transformation
        model = BertMaskModelWithMLP(args, bert_version=args.bert_version)
        model_params.append('with_MLP')
        if args.nonlinear:
            model_params.append('nonlinear')
        else:
            model_params.append('linear')
        if "base" in args.bert_version:
            unfreeze_layers = ['layer.11', 'layer.10', 'layer.9', 'trans1', 'trans2']
        else:
            unfreeze_layers = ['layer.23', 'layer.22', 'layer.21', 'trans1', 'trans2']
    else:
        model = BertMaskModel(args, bert_version=args.bert_version)
        if "base" in args.bert_version:
            unfreeze_layers = ['layer.11', 'layer.10', 'layer.9']
        else:
            unfreeze_layers = ['layer.23', 'layer.22', 'layer.21']

    tuned_parameters = [
        {'params': [param for name, param in model.named_parameters() if any(ul in name for ul in unfreeze_layers)]}]

    optimizer = optim.AdamW(tuned_parameters, lr=args.lr)
    model_file = os.path.join(model_path, "_".join(model_params) + "_last3L.pt")
    early_stopping = EarlyStopping(patience=10, verbose=False, path=model_file, delta=1e-10)
    if args.lr_schedule == 'linear':
        scheduler = get_linear_schedule_with_warmup(optimizer, len(train_loader) * 2, int(len(train_loader) * 1000))
    else:
        scheduler = get_cosine_schedule_with_warmup(optimizer, len(train_loader) * 2, int(len(train_loader) * 1000),
                                                    num_cycles=1.5)

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)  # random batch training seed (shuffle) to ensure reproducibility

    use_gpu = False
    if torch.cuda.is_available():
        use_gpu = True
        n_gpu = torch.cuda.device_count()
        if n_gpu > 1:
            logging.info("use multiple GPUs")
            model = torch.nn.DataParallel(model)
            # torch.cuda.manual_seed_all(args.seed)
        model.to("cuda")

    model.train()
    logging.info("start training ...")
    for epoch in trange(1000, desc="Epoch"):
        train_loss = 0
        for step, batch in enumerate(tqdm(train_loader, desc="Iteration")):
            idx1 = batch[0][:, 0]
            idx2 = batch[0][:, 1]
            s1 = np.array(sentences)[idx1]
            s2 = np.array(sentences)[idx2]
            w1 = np.array(word_list)[idx1]
            w2 = np.array(word_list)[idx2]

            in_1 = tokenize_mask(s1, w1, args.max_seq_len, bert_tokenizer)
            in_2 = tokenize_mask(s2, w2, args.max_seq_len, bert_tokenizer)

            token_ids_1, input_mask_1, indices_1 = convert_tuple_to_tensors(in_1, use_gpu)
            token_ids_2, input_mask_2, indices_2 = convert_tuple_to_tensors(in_2, use_gpu)

            optimizer.zero_grad()
            loss = model(token_ids_1, input_mask_1, indices_1, token_ids_2, input_mask_2, indices_2, idx1)
            if isinstance(model, torch.nn.DataParallel):
                loss = loss.mean()
            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()

            del in_1, in_2
            del token_ids_1, input_mask_1, indices_1
            del token_ids_2, input_mask_2, indices_2
            del loss
            gc.collect()

        train_loss /= (step + 1)
        torch.cuda.empty_cache()

        # validation
        model.eval()
        dev_loss = 0
        for step, batch in enumerate(tqdm(dev_loader, desc="dev")):
            idx1 = batch[0][:, 0]
            idx2 = batch[0][:, 1]
            s1 = np.array(sentences)[idx1]
            s2 = np.array(sentences)[idx2]
            w1 = np.array(word_list)[idx1]
            w2 = np.array(word_list)[idx2]

            in_1 = tokenize_mask(s1, w1, args.max_seq_len, bert_tokenizer)
            in_2 = tokenize_mask(s2, w2, args.max_seq_len, bert_tokenizer)

            token_ids_1, input_mask_1, indices_1 = convert_tuple_to_tensors(in_1, use_gpu)
            token_ids_2, input_mask_2, indices_2 = convert_tuple_to_tensors(in_2, use_gpu)

            loss = model(token_ids_1, input_mask_1, indices_1, token_ids_2, input_mask_2, indices_2, idx1)
            if isinstance(model, torch.nn.DataParallel):
                loss = loss.mean()
            dev_loss += loss.item()

            del in_1, in_2
            del token_ids_1, input_mask_1, indices_1
            del token_ids_2, input_mask_2, indices_2
            del loss
            gc.collect()

        dev_loss /= (step + 1)
        model.train()

        logging.info("Epoch: %d | train loss: %.4f | dev loss: %.4f ", epoch + 1, train_loss, dev_loss)

        torch.cuda.empty_cache()

        if epoch >= 5:
            early_stopping(dev_loss, model)

        if early_stopping.early_stop:
            logging.info("Early stopping. Model trained")
            break

    torch.cuda.empty_cache()

    # extract mention vectors from fine-tuned BERT model
    # model.to("cpu")
    model.load_state_dict(torch.load(model_file))
    if use_gpu:
        model.to('cuda')

    logging.info("extract mention vectors from tuned BERT model")

    if not os.path.exists(os.path.join(args.out_path, args.dataset, '_'.join(model_params) + '_last3L')):
        os.makedirs(os.path.join(args.out_path, args.dataset, '_'.join(model_params) + '_last3L'))

    cnt = 0
    bert_mv = torch.tensor([])
    model.eval()
    for w in words:
        if cnt % 500 == 0:
            logging.info(str(cnt) + "/" + str(len(words)) + " processed.")

        # build dataloader
        wi = np.where(np.array(word_list) == w)[0]
        wi = torch.tensor(wi, dtype=torch.long)
        dataset = TensorDataset(wi)
        loader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=False)

        emb_wi = torch.tensor([])
        for batch in loader:
            wi_batch = batch[0]
            if wi_batch.size()[0] == 1:
                sentences_batch = [np.array(sentences)[wi_batch]]
                words_batch = [np.array(word_list)[wi_batch]]
            else:
                sentences_batch = np.array(sentences)[wi_batch]
                words_batch = np.array(word_list)[wi_batch]

            input_tuple = tokenize_mask(sentences_batch, words_batch, args.max_seq_len, bert_tokenizer)
            token_ids, input_masks, indices = convert_tuple_to_tensors(input_tuple, use_gpu)
            if isinstance(model, torch.nn.DataParallel):
                emb = model.module.get_bert_emb(token_ids, input_masks, indices).detach().cpu()
            else:
                emb = model.get_bert_emb(token_ids, input_masks, indices).detach().cpu()
            emb_wi = torch.cat((emb_wi, emb))
            del emb
            del token_ids, input_masks, indices
            torch.cuda.empty_cache()

        torch.save(emb_wi, os.path.join(args.out_path, args.dataset, '_'.join(model_params) + '_last3L', w + ".pt"))
        bert_mv = torch.cat((bert_mv, emb_wi))
        cnt += 1
    torch.save(bert_mv, os.path.join(args.out_path, args.dataset, '_'.join(model_params) + '_last3L.pt'))

    return bert_mv, word_list
