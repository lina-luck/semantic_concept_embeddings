# util function for finetuning bert model by distant supervision from conceptNet
import logging
import os
import sys
project_path = os.path.split(os.path.abspath(os.path.realpath(__file__)))[0] + "/../"
sys.path.append(os.path.abspath(project_path))

from src.utils import load_text, build_positive_triple, dataloader
from sklearn.model_selection import train_test_split
from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup, AutoTokenizer
from transformers.optimization import AdamW
from tqdm import tqdm, trange
from src.models import BertMaskModelWithMLP, BertMaskModel
from src.finetune_utils import convert_tuple_to_tensors
from src.early_stop import *


def tokenize_mask(properties, prompts, bert_tokenizer, max_seq_len):
    """
    Tokenize single prompt for BERT Masked Model
    :param properties: list of properties
    :param prompts: list of prompts
    :param bert_tokenizer: tokenizer
    :param max_seq_len: max sequence length
    :return:
    tuple of token ids, mask ids and indices of mask
    """
    cls_token = bert_tokenizer.convert_ids_to_tokens(bert_tokenizer.cls_token_id)  # "[CLS]"
    sep_token = bert_tokenizer.convert_ids_to_tokens(bert_tokenizer.sep_token_id)  # "[SEP]"
    mask_token = bert_tokenizer.convert_ids_to_tokens(bert_tokenizer.mask_token_id)  # "[MASK]"
    token_ids = []
    input_masks = []
    indices = []
    for prop, prompt in zip(properties, prompts):
        prompt = prompt.replace("[MASK]", mask_token)
        tokens = bert_tokenizer.tokenize(prompt)

        if ' <mask>' in tokens:
            tokens[tokens.index(' <mask>')] = mask_token

        if len(tokens) > max_seq_len - 2:
            if tokens.index(mask_token) >= max_seq_len - 3:  # left part of [MASK] is longer than max_seq_len
                tokens = tokens[(tokens.index(mask_token) - (max_seq_len-3) // 2):]
            else:
                tokens = tokens[:(max_seq_len - 2)]
        tokens = [cls_token] + tokens + [sep_token]
        idx = tokens.index(mask_token)
        t_id = bert_tokenizer.convert_tokens_to_ids(tokens)
        padding = [0] * (max_seq_len - len(t_id))
        i_mask = [1] * len(t_id) + padding
        t_id += padding

        token_ids.append(t_id)
        input_masks.append(i_mask)
        indices.append(idx)
    return token_ids, input_masks, indices


def finetune_with_ds(args):
    '''
    main function for fine-tuning with distant supervision from conceptNet
    '''
    model_path = os.path.join('finetuned_model_with_ds')
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    pos_triple_path = os.path.join('pos_triple')
    if not os.path.exists(pos_triple_path):
        os.makedirs(pos_triple_path)

    # loss name
    if args.bert_version.endswith('/'):
        args.bert_version = args.bert_version[:-1]

    model_params = [os.path.basename(args.bert_version), args.loss]

    if args.loss == "infonce":
        model_params.append(str(args.tau))
    elif args.loss == "triple":
        model_params.append(str(args.margin))

    model_params += [str(args.lr), str(args.batch_size), str(args.weight_decay)]
    model_file = os.path.join(model_path, "_".join(model_params) + ".pt")

    logging.info(str(args))

    # 1. load properties
    logging.info("load properties")
    properties_ = load_text(args.prop_file)
    logging.info("total property number is " + str(len(properties_)))

    if 0 < args.prop_num < len(properties_):
        logging.info("use top " + str(args.prop_num) + " frequent properties")
        properties_ = properties_[:args.prop_num]

    # 2. build positive triples (prop, prompt1, prompt2)
    prompt_pairs, prop_prompt_idx, properties = build_positive_triple(properties_, args.ppt_path, pos_triple_path)
    del properties_

    logging.info("number of positive pairs is " + str(len(prop_prompt_idx)))

    # 3. build data loader
    train_pairs, dev_pairs = train_test_split(prop_prompt_idx, train_size=0.8)
    train_labels = np.array(train_pairs)[:, 0]
    train_loader = dataloader(train_pairs, args.batch_size, is_train=True, labels=train_labels)
    dev_loader = dataloader(dev_pairs, args.batch_size)

    del prop_prompt_idx
    del train_pairs
    del dev_pairs

    # build model
    tokenizer = AutoTokenizer.from_pretrained(args.bert_version)
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

    optimizer = AdamW(tuned_parameters, lr=args.lr)
    early_stopping = EarlyStopping(patience=10, verbose=False, path=model_file, delta=1e-10)

    if args.lr_schedule == 'linear':
        scheduler = get_linear_schedule_with_warmup(optimizer, len(train_loader) * 2, len(train_loader) * 1000)
    else:
        scheduler = get_cosine_schedule_with_warmup(optimizer, len(train_loader) * 2, len(train_loader) * 1000, num_cycles=1.5)

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)  # random batch training seed (shuffle) to ensure reproducibility

    use_gpu = False
    if torch.cuda.is_available():
        use_gpu = True
        logging.info("use gpu")
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
            prop_idx = batch[0][:, 0]
            prompt_pair_idx = batch[0][:, 1]

            prop_batch = np.array(properties)[prop_idx]
            prompt_pair_batch = np.array(prompt_pairs)[prompt_pair_idx]

            if prompt_pair_batch.ndim == 1:
                prompt_pair_batch = prompt_pair_batch.reshape((-1, prompt_pair_batch.shape[0]))

            in_1 = tokenize_mask(prop_batch, prompt_pair_batch[:, 0], tokenizer, args.max_seq_len)
            in_2 = tokenize_mask(prop_batch, prompt_pair_batch[:, 1], tokenizer, args.max_seq_len)

            token_ids_1, input_mask_1, indices_1 = convert_tuple_to_tensors(in_1, use_gpu)
            token_ids_2, input_mask_2, indices_2 = convert_tuple_to_tensors(in_2, use_gpu)

            optimizer.zero_grad()
            loss = model(token_ids_1, input_mask_1, indices_1, token_ids_2, input_mask_2, indices_2, prop_idx)
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

        train_loss /= (step + 1)
        torch.cuda.empty_cache()

        # validation
        model.eval()
        dev_loss = 0
        for step, batch in enumerate(tqdm(dev_loader, desc="dev")):
            prop_idx = batch[0][:, 0]
            prompt_pair_idx = batch[0][:, 1]

            prop_batch = np.array(properties)[prop_idx]
            prompt_pair_batch = np.array(prompt_pairs)[prompt_pair_idx]

            if prompt_pair_batch.ndim == 1:
                prompt_pair_batch = prompt_pair_batch.reshape((-1, prompt_pair_batch.shape[0]))

            in_1 = tokenize_mask(prop_batch, prompt_pair_batch[:, 0], tokenizer, args.max_seq_len)
            in_2 = tokenize_mask(prop_batch, prompt_pair_batch[:, 1], tokenizer, args.max_seq_len)

            token_ids_1, input_mask_1, indices_1 = convert_tuple_to_tensors(in_1, use_gpu)
            token_ids_2, input_mask_2, indices_2 = convert_tuple_to_tensors(in_2, use_gpu)

            loss = model(token_ids_1, input_mask_1, indices_1, token_ids_2, input_mask_2, indices_2, prop_idx)
            if isinstance(model, torch.nn.DataParallel):
                loss = loss.mean()
            dev_loss += loss.item()

            del in_1, in_2
            del token_ids_1, input_mask_1, indices_1
            del token_ids_2, input_mask_2, indices_2
            del loss

        dev_loss /= (step + 1)
        model.train()

        logging.info("Epoch: %d | train loss: %.5f | dev loss: %.5f ", epoch + 1, train_loss, dev_loss)

        torch.cuda.empty_cache()

        if epoch >= 5:
            early_stopping(dev_loss, model)

        if early_stopping.early_stop:
            logging.info("Early stopping. Model trained")
            break
