import argparse
import sys
import os

project_path = os.path.split(os.path.abspath(os.path.realpath(__file__)))[0] + "/../"
sys.path.append(os.path.abspath(project_path))
from src.utils import *
from src.cnn_utils import train, test
from src.svm_utils import train_svc, test_svc
from src.run_cl import preprocess_transformed_vectors_for_svm, preprocess_transformed_vectors_for_cnn, load_csv
import torch


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("-data_dir", help="data path", default="../data")
    parser.add_argument("-dataset", help="dataset", default="battig")

    parser.add_argument("-filter_path", help="path of knn word file", default="../mv_knn")
    parser.add_argument("-mv_knn_file", help="file name of mv knn file", default="mv_knn/battig/cos_20.npy")
    parser.add_argument("-emb_path", help="embedding path or file", default="cl_out/battig/mv_knn_10_cos_64_2e-05_32_0.05_0.001.pt")
    parser.add_argument("-k", help="neighbor number", type=int, default=2)
    parser.add_argument("-res_path", help="result path", default="results_cl")
    parser.add_argument("-vec_num_per_word", help="vec_num_per_word", default=1, type=int)
    parser.add_argument("-use_mask", help="whether the model is maskedLM or not", default=False, action="store_true")
    parser.add_argument("-filter", help="whether to use filtering strategy", default=False, action="store_true")
    parser.add_argument("-cls", help="run cnn or svm", default='svm', choices=["cnn", "svm"])

    args = parser.parse_args()

    log_file_path = init_logging_path("log", "classifier")
    logging.basicConfig(filename=log_file_path,
                        level=10,
                        format='%(asctime)s %(levelname)s %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p')

    logging.info(str(args))

    result_path = os.path.join(os.path.abspath(project_path), args.res_path, args.dataset)
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    is_baseline_embedding = False
    # 1. load embedding
    logging.info("load embeddings")
    if os.path.isfile(args.emb_path):  # if emb_path is a file
        file_format = os.path.splitext(args.emb_path)[-1]
        if file_format in [".gz", ".txt", ".pickle"]:  # baseline emb file
            embeddings = load_baseline_emb(args.emb_path)
            is_baseline_embedding = True
        else:
            embeddings_org = torch.load(args.emb_path)
            word_list = get_word_list(args.mv_knn_file)
    elif os.path.isdir(args.emb_path):  # if emb_path is a directory
        words = []
        with open(os.path.join(args.data_dir, args.dataset, args.dataset + '.txt'), 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip().lower() == "french_horn":
                    continue
                words.append(line.strip().lower())
        embeddings_org = torch.tensor([])
        word_list = []
        for w in words:
            if w + ".pt" in os.listdir(args.emb_path):
                emb_w = torch.load(os.path.join(args.emb_path, w + ".pt"))
            elif w + ".csv" in os.listdir(args.emb_path):
                emb_w = load_csv(os.path.join(args.emb_path, w + ".csv"))
                emb_w = torch.tensor(emb_w)
            embeddings_org = torch.cat((embeddings_org, emb_w))
            word_list.extend([w] * emb_w.shape[0])

    # load data
    logging.info("load data")
    pos_train = load_prop_instances(os.path.join(args.data_dir, args.dataset, f'pos_train_data.txt'))[0]
    pos_test = load_prop_instances(os.path.join(args.data_dir, args.dataset, f"pos_test_data.txt"))[0]
    neg_test = load_prop_instances(os.path.join(args.data_dir, args.dataset, f"neg_test_data.txt"))[0]
    pos_valid = load_prop_instances(os.path.join(args.data_dir, args.dataset, f"pos_valid_data.txt"))[0]
    neg_valid = load_prop_instances(os.path.join(args.data_dir, args.dataset, f"neg_valid_data.txt"))[0]
    neg_train = load_prop_instances(os.path.join(args.data_dir, args.dataset, f"neg_train_data.txt"))[0]

    if args.cls == 'cnn':
        # run cnn
        embeddings = preprocess_transformed_vectors_for_cnn(embeddings_org, word_list, args.vec_num_per_word)
        # parameters for cnn training
        cnn_out_dim = [64]  # [128, 64]
        cnn_kernel_size = [1, 2, 5]
        cnn_batch_size = [16]
        pool_way = ['avg', 'max']
        args.cnn_stride = 1
        args.cnn_padding = 0
        cnn_lr = [1e-3]
        args.cnn_in_dim = embeddings_org.size()[1]
        args.weight_decay = 1e-3

        res_all_prop = []
        cnt = 1
        for prop in pos_train:
            best_res_prop = [0] * 4
            best_pram_prop = None
            logging.info(str(cnt) + ', ' + prop)
            if args.filter:  # if filtering
                if args.use_mask:
                    knn_file = os.path.join(args.filter_path, 'knn_mask_large_' + args.dataset + '_' +
                                            prop + '_64' + '.txt')
                else:
                    knn_file = os.path.join(args.filter_path, 'knn_nomask_large_' + args.dataset + '_' +
                                            prop + '_64' + '.txt')
                logging.info("loading neighbors")
                knn = load_neighbors(knn_file, args.k)

                logging.info("filtering")
                remain_word_idx = filter_strategy_rosv(knn[:, :args.k + 1])

                train_data, train_label = process_filtered_data_for_cnn(embeddings,
                                                                uniq_word_ordered(word_list),
                                                                pos_train[prop], neg_train[prop], remain_word_idx)
                valid_data, valid_label = process_filtered_data_for_cnn(embeddings,
                                                                uniq_word_ordered(word_list),
                                                                pos_valid[prop], neg_valid[prop], remain_word_idx)
                test_data, test_label = process_filtered_data_for_cnn(embeddings,
                                                              uniq_word_ordered(word_list),
                                                              pos_test[prop], neg_test[prop], remain_word_idx)
            else:
                train_data, train_label = process_data_for_cnn(embeddings, uniq_word_ordered(word_list),
                                                       pos_train[prop], neg_train[prop])
                valid_data, valid_label = process_data_for_cnn(embeddings, uniq_word_ordered(word_list),
                                                       pos_valid[prop], neg_valid[prop])
                test_data, test_label = process_data_for_cnn(embeddings, uniq_word_ordered(word_list),
                                                     pos_test[prop], neg_test[prop])

            for args.cnn_out_dim in cnn_out_dim:
                for args.cnn_kernel_size in cnn_kernel_size:
                    for args.cnn_bsz in cnn_batch_size:
                        for args.cnn_pool_way in pool_way:
                            for args.cnn_lr in cnn_lr:
                                cnn_params = ['cnn_out_dim = ' + str(args.cnn_out_dim),
                                              'cnn_in_dim = ' + str(args.cnn_in_dim),
                                              'cnn_kernel_size = ' + str(args.cnn_kernel_size),
                                              'cnn_batch_size = ' + str(args.cnn_bsz),
                                              'cnn_pool_way = ' + args.cnn_pool_way,
                                              'cnn_lr = ' + str(args.cnn_lr)]

                                logging.info(str(cnt) + ', ' + prop)

                                model, th = train(args, train_data, valid_data, train_label, valid_label)
                                res_prop = test(test_data, test_label, model, th, args.cnn_bsz)

                                if best_res_prop[0] * best_res_prop[-1] < res_prop[0] * res_prop[-1]:
                                    best_res_prop = res_prop
                                    best_pram_prop = cnn_params

            res_all_prop.append(best_res_prop)
            if best_pram_prop is not None:
                logging.info(', '.join(best_pram_prop))
            logging.info(str(best_res_prop))
            cnt += 1

        res_mean = np.mean(np.array(res_all_prop), axis=0)

        results_str = str(args) + '\n' + ": map = " + str(res_mean[0]) + ", f1 = " + str(res_mean[-1]) + "\n"

        csv_file = os.path.join(result_path, "cnn_" + str(args.cnn_bsz) + "_" + str(args.cnn_lr) + '.csv')
        write_csv(csv_file, list(pos_train.keys()), res_all_prop)

        logging.info("cnn best results are: map = " + str(res_mean[0]) + ", f1 = " + str(res_mean[-1]))

        with open(os.path.join(result_path, args.dataset + ".txt"), 'a+', encoding="utf-8") as f:
            f.write(results_str)

    elif args.cls == 'svm':
        # run svm
        logging.info("running linear svm")
        if not is_baseline_embedding:
            embeddings = preprocess_transformed_vectors_for_svm(embeddings_org, word_list)
        res_all_prop = []
        cnt = 1
        for prop in pos_train:
            logging.info(str(cnt) + ', ' + prop)
            if args.filter:
                if is_baseline_embedding:
                    raise ValueError("args.filter can't be True when baseline embedding is used")
                if args.use_mask:
                    knn_file = os.path.join(args.filter_path, 'knn_mask_large_' + args.dataset + '_' +
                                    prop + '_64' + '.txt')
                else:
                    knn_file = os.path.join(args.filter_path, 'knn_nomask_large_' + args.dataset + '_' +
                                    prop + '_64' + '.txt')
                logging.info("loading neighbors")
                knn = load_neighbors(knn_file, args.k)

                logging.info("filtering")
                remain_word_idx = filter_strategy_rosv(knn[:, :args.k + 1])

                embeddings = preprocess_filter_transformed_vectors_for_svm(embeddings_org, word_list, remain_word_idx)

            train_data, train_label = process_data_for_svm(embeddings, pos_train[prop], neg_train[prop])
            valid_data, valid_label = process_data_for_svm(embeddings, pos_valid[prop], neg_valid[prop])
            test_data, test_label = process_data_for_svm(embeddings, pos_test[prop], neg_test[prop])

            svm, th = train_svc(train_data, valid_data, train_label, valid_label, "linear",
                            cv=min(3, len(pos_train[prop])))
            rr = test_svc(test_data, test_label, svm, th)
            logging.info(str(cnt) + ', ' + prop + ': map = ' + str(rr[0]) + ', f1 = ' + str(rr[-1]))
            res_all_prop.append(rr)
            cnt += 1

        res_mean = np.mean(np.array(res_all_prop), axis=0)
        print(args.dataset, res_mean[-1])
        results_str = str(args) + '\nLinear svm\n' + ": map = " + str(res_mean[0]) + ", f1 = " + str(res_mean[-1]) + "\n\n\n"

        csv_file = os.path.join(result_path, 'svm_linear.csv')
        write_csv(csv_file, list(pos_train.keys()), res_all_prop)

        logging.info("svm best results are: map = " + str(res_mean[0]) + ", f1 = " + str(res_mean[-1]))

        with open(os.path.join(result_path, args.dataset + ".txt"), 'a+', encoding="utf-8") as f:
            f.write(results_str)

    logging.info("done")






