## util file for running cnn
import os
import sys
project_path = os.path.split(os.path.abspath(os.path.realpath(__file__)))[0] + "/../"
sys.path.append(os.path.abspath(project_path))
from src.utils import *
from src.models import CNN_Classifier
from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import average_precision_score, f1_score
from torch.utils.data import RandomSampler, SequentialSampler, TensorDataset, DataLoader


# build cnn dataloader
def cnn_dataloader(data, label, batch_size, mode="train"):
    data = torch.tensor(data, dtype=torch.float)
    label = torch.tensor(label, dtype=torch.float)
    dataset = TensorDataset(data, label)
    if mode == "train":
        sampler = RandomSampler(dataset)
    else:
        sampler = SequentialSampler(dataset)
    loader = DataLoader(dataset=dataset, batch_size=batch_size, sampler=sampler)
    return loader


# train cnn, return trained cnn model
def train(args, train_data, valid_data, train_label, valid_label):
    # loader
    train_loader = cnn_dataloader(train_data, train_label, args.cnn_bsz, "train")
    valid_loader = cnn_dataloader(valid_data, valid_label, args.cnn_bsz, "dev")

    _, vec_num_per_word, emb_dim = train_data.shape
    model = CNN_Classifier(args.cnn_in_dim, args.cnn_out_dim, args.cnn_kernel_size, vec_num_per_word, args.cnn_stride,
                           args.cnn_padding, args.cnn_pool_way)
    optimizer = Adam(model.parameters(), weight_decay=args.weight_decay, lr=args.cnn_lr)
    scheduler = ReduceLROnPlateau(optimizer, 'min')

    if not os.path.exists(os.path.join("cnn_model", args.dataset)):
        os.makedirs(os.path.join("cnn_model", args.dataset))
    cnn_param = [str(args.cnn_bsz), str(args.cnn_in_dim), str(args.cnn_out_dim), str(args.cnn_kernel_size),
                 args.cnn_pool_way]
    cnn_m_file = os.path.join("cnn_model", args.dataset, "_" + '_'.join(cnn_param) + ".pt")
    early_stopping = EarlyStopping(patience=10, path=cnn_m_file)

    use_gpu = False
    if torch.cuda.is_available():
        use_gpu = True
        model.to('cuda')

    model.train()
    for epoch in trange(200, desc="Epoch"):
        tr_loss = 0
        for step, batch in enumerate(tqdm(train_loader, desc="Iteration")):
            X, y = batch
            if use_gpu:
                X = X.to("cuda")
                y = y.to("cuda")

            optimizer.zero_grad()
            logits, loss = model(X, y)
            loss.backward()
            optimizer.step()

            tr_loss += loss.item()
            del loss
            del X
            del y

        tr_loss /= (step + 1)

        loss_dev, y_true, y_pred = evaluate(valid_loader, model, use_gpu)

        dev_map = -average_precision_score(y_true, y_pred)
        if (epoch + 1) % 10000 == 0:
            logging.info("Epoch: %d | train loss: %.4f | valid loss: %.4f | valid map: %.4f ",
                         epoch + 1, tr_loss, loss_dev, -dev_map)
        scheduler.step(dev_map)
        model.train()
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
        if epoch > 5:
            early_stopping(dev_map, model)

        if early_stopping.early_stop:
            logging.info("Early Stopping. Model trained.")
            break

    # choose threshold
    model.load_state_dict(torch.load(cnn_m_file))
    _, y_true, y_score = evaluate(valid_loader, model, use_gpu)
    th, _ = optimal_threshold(y_true, y_score)
    return model, th


# compute loss
def evaluate(dataloader, model, use_gpu=False):
    # dataloader: cnn dataloader
    # model: cnn model
    # use_gpu: whether to use gpu
    model.eval()
    y_true = []
    y_pred = []
    loss = 0
    for step, batch in enumerate(tqdm(dataloader, desc="Evaluation")):
        X, y = batch
        if use_gpu:
            model = model.to("cuda")
            X = X.to("cuda")
            y = y.to("cuda")

        logits, batch_loss = model(X, y)
        y_score = model.sigmoid_fn(logits)
        y_pred.extend(y_score.data.cpu().clone().numpy())
        y_true.extend(y.data.cpu().clone().numpy())

        loss += batch_loss.item()
        del X
        del y
        del batch_loss
    loss /= (step + 1)
    return loss, np.array(y_true), np.array(y_pred)


# test cnn, return average precision, precision, recall, f1
def test(test_data, test_label, model, th, batch_size):
    # test_data: embedding of test data
    # test_label: test label
    # model: trained cnn model
    # th: threshold for classification
    # batch_size: batch size
    test_loader = cnn_dataloader(test_data, test_label, batch_size)

    use_gpu = False
    if torch.cuda.is_available():
        use_gpu = True
    _, y_true, y_score = evaluate(test_loader, model, use_gpu)

    y_pred = np.zeros_like(y_true, dtype=int)
    idx = np.where(y_score >= th)[0]
    y_pred[idx] = 1

    ap = round(average_precision_score(y_true, y_score), 4)
    pre, rec, f1 = pre_rec_f1(y_true, y_pred)
    del model
    return [ap, pre, rec, f1]


# main function to run cnn classifier
def run_cnn(args, embeddings, all_words, pos_train, neg_train, pos_test, neg_test, pos_valid, neg_valid):
    # embeddings: embedding of all words
    # all_words: words corresponding to embedding
    # pos_train: positive training words
    # neg_train: negative training words
    # pos_test, neg_test, pos_valid, neg_valid: similar with pos_train and neg_train

    logging.info('start to train and test cnn')
    # data preprocessing
    pos_train_data = init_word_embedding(embeddings, all_words, pos_train)
    neg_train_data = init_word_embedding(embeddings, all_words, neg_train)
    pos_test_data = init_word_embedding(embeddings, all_words, pos_test)
    neg_test_data = init_word_embedding(embeddings, all_words, neg_test)
    pos_valid_data = init_word_embedding(embeddings, all_words, pos_valid)
    neg_valid_data = init_word_embedding(embeddings, all_words, neg_valid)

    train_data = torch.cat((pos_train_data, neg_train_data), dim=0)
    train_label = np.array(
            [1] * pos_train_data.shape[0] + [0] * neg_train_data.shape[0])
    test_data = torch.cat((pos_test_data, neg_test_data), dim=0)
    test_label = np.array([1] * pos_test_data.shape[0] + [0] * neg_test_data.shape[0])
    valid_data = torch.cat((pos_valid_data, neg_valid_data), dim=0)
    valid_label = np.array(
            [1] * pos_valid_data.shape[0] + [0] * neg_valid_data.shape[0])
    del pos_train_data
    del neg_train_data
    del pos_test_data
    del neg_test_data
    del pos_valid_data
    del neg_valid_data

    model, th = train(args, train_data, valid_data, train_label, valid_label)
    rr = test(test_data, test_label, model, th, args.batch_size)
    return rr