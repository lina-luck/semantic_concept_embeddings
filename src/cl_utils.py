## util file for contrastive learning model
import logging
import os
import sys
project_path = os.path.split(os.path.abspath(os.path.realpath(__file__)))[0] + "/../"
sys.path.append(os.path.abspath(project_path))
from src.utils import *
from src.models import *
from src.find_knn import load_baseline_emb

# run contrastive learning model
def run(args):
    model_path = os.path.join("cl_models", args.dataset)
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    pos_pair_path = os.path.join('pos_pair', args.dataset)
    if not os.path.exists(pos_pair_path):
        os.makedirs(pos_pair_path)

    method_dict = {1: "mv_knn", 2: "word_knn_js", 3: "word_knn_pi"}
    model_params = [method_dict[args.method]]
    if args.method == 1:
        model_params.extend([str(args.k)])
        if args.use_cosine:
            model_params.append("cos")
        else:
            model_params.append("L2")
    elif args.method in [2, 3]:
        model_params.extend([str(args.k), str(args.olp_th)])

    model_params.extend([str(args.out_dim), str(args.lr), str(args.batch_size), str(args.tau), str(args.weight_decay)])
    model_file = os.path.join(model_path, "_".join(model_params) + ".pt")

    logging.info(str(args))
    # 1. load words
    if args.w_file is not None:
        words = []
        with open(args.w_file, 'r', encoding="utf-8") as f:
            for line in f:
                words.append(line.strip().replace(' ', '_'))
        with open("../data/battig/battig.txt", 'r', encoding="utf-8") as f:
            for line in f:
                if line.strip() not in words:
                    words.append(line.strip().replace(' ', '_'))
    else:
        words = [f.split('.')[0] for f in os.listdir(args.mv_path)]

    # 2. load mention vectors
    logging.info("load in mention vectors")
    men_vec = torch.load(args.mv_path).numpy()
    word_list = words
    del words
    # 3. positive pairs of mv index
    logging.info("build data loader")
    mv_idx_pairs = build_positive_pair(men_vec, word_list, method=args.method, k=args.k, mv_knn_file=args.mv_knn_file,
                                           use_cosine=args.use_cosine, word_knn_file=args.word_knn_file,
                                           olp_th=args.olp_th, out_path=pos_pair_path)

    train_pairs, dev_pairs = train_test_split(mv_idx_pairs, train_size=0.8)
    logging.info("train pair num = " + str(len(train_pairs)) + ", dev pair num = " + str(len(dev_pairs)))
    train_labels = None
    if args.method == 1:
        train_labels = np.array(train_pairs)[:, 0]
    train_loader = dataloader(train_pairs, args.batch_size, is_train=True, labels=train_labels)
    dev_loader = dataloader(dev_pairs, args.batch_size)
    del mv_idx_pairs
    del train_pairs
    del dev_pairs

    # 4. prepare model, optimizer, and so on
    model = ContrastiveLearningModel2(args.in_dim, args.hidden_dim, args.out_dim, args.tau, args.nonlinear)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.lr_schedule == "cosine":
        scheduler = get_cosine_schedule_with_warmup(optimizer, len(train_loader) * 2, int(len(train_loader) * 1000))
    else:
        scheduler = get_linear_schedule_with_warmup(optimizer, len(train_loader) * 2, int(len(train_loader) * 1000))
    logging.info("num of steps in an epoch: " + str(len(train_loader)))
    early_stop = EarlyStopping(patience=10, path=model_file, delta=1e-10)

    use_gpu = False
    if torch.cuda.is_available():
       use_gpu = True
       logging.info("use gpu")
       if torch.cuda.device_count() > 1:
           logging.info("use multiple gpu")
           model = torch.nn.DataParallel(model)
       model.to('cuda')

    logging.info("training model")
    men_vec = torch.tensor(men_vec, dtype=torch.float)

    model.train()
    for epoch in trange(1000, desc="Epoch"):
        tr_loss = 0
        for step, batch in enumerate(tqdm(train_loader, desc="Iteration")):
            idx1 = batch[0][:, 0]
            idx2 = batch[0][:, 1]
            x1 = men_vec[idx1]
            x2 = men_vec[idx2]
            if use_gpu:
                x1 = x1.to("cuda")
                x2 = x2.to("cuda")

            optimizer.zero_grad()
            loss = model(x1, x2, args.use_hard_pair, idx1)
            if isinstance(model, torch.nn.DataParallel):
                loss = loss.mean()
            loss.backward()
            optimizer.step()
            scheduler.step()
            del idx1
            del idx2
            del x1
            del x2
            tr_loss += loss.item()
            del loss

        tr_loss /= (step + 1)

        # validation
        model.eval()
        dev_loss = 0
        for step, batch in enumerate(tqdm(dev_loader, desc="dev")):
            idx1 = batch[0][:, 0]
            idx2 = batch[0][:, 1]
            x1 = men_vec[idx1]
            x2 = men_vec[idx2]
            if use_gpu:
                x1 = x1.to("cuda")
                x2 = x2.to("cuda")

            loss = model(x1, x2, args.use_hard_pair, idx1)
            del idx1
            del idx2
            del x1
            del x2
            if isinstance(model, torch.nn.DataParallel):
                loss = loss.mean()
            dev_loss += loss.item()
            del loss
        dev_loss /= (step + 1)
        model.train()

        logging.info("Epoch: %d | train loss: %.4f | dev loss: %.4f ", epoch + 1, tr_loss, dev_loss)
        if epoch >= 5:
            early_stop(dev_loss, model)

        torch.cuda.empty_cache()

        if early_stop.early_stop:
            logging.info("Early Stopping. Model trained.")
            break

    model.to("cpu")
    model.load_state_dict(torch.load(model_file, map_location=torch.device('cpu')))
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    transformed_vectors = model.transformation(men_vec)

    outfile = os.path.join(args.out_path, args.dataset, "_".join(model_params) + ".pt")
    torch.save(transformed_vectors, outfile)
    return transformed_vectors, word_list
