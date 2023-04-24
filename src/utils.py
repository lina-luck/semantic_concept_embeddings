import os
import csv
import logging
import time

import torch
import numpy as np
from faiss import normalize_L2, IndexFlatL2, IndexFlatIP
import bz2, json, contextlib
from torch.utils.data import TensorDataset, DataLoader
from pytorch_metric_learning.samplers import MPerClassSampler
from sklearn.model_selection import train_test_split
from src.early_stop import *
from torch import optim
from transformers import get_constant_schedule_with_warmup, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from tqdm import tqdm, trange
from sklearn.metrics import precision_score, recall_score, f1_score
from scipy.optimize import brute
import pickle
from gensim.models.keyedvectors import KeyedVectors
from itertools import combinations
from multiprocessing.dummy import Pool, freeze_support


def process_data_for_cnn(embeddings, all_words, pos_words, neg_words):
    """
    get embedding for positive and negative words for cnn
    :param embeddings:
    :param all_words:
    :param pos_words:
    :param neg_words:
    :return:
    """
    pos_data = init_word_embedding_for_cnn(embeddings, all_words, pos_words)
    neg_data = init_word_embedding_for_cnn(embeddings, all_words, neg_words)

    data = torch.cat((pos_data, neg_data), dim=0)
    label = np.array(
            [1] * pos_data.shape[0] + [0] * neg_data.shape[0])
    return data, label


# load sentences
def load_sentences(words, sent_path, word_list):
    # words: words whose sentences will be loaded
    # sent_path: input path of sentence files
    # word_list: word list
    sentences = []
    for w in words:
        w_sent = []
        with open(os.path.join(sent_path, w + ".txt"), "r", encoding="utf-8") as f:
            for line in f:
                sent = line.strip().lower()
                w_sent.append(sent)
        assert len(w_sent) == np.where(np.array(word_list)==w)[0].shape[0]
        sentences.extend(w_sent)
    assert len(sentences) == len(word_list)
    return sentences


## build log file
def init_logging_path(log_path, file_name):
    dir_log  = os.path.join(log_path,f"{file_name}/")
    if os.path.exists(dir_log) and os.listdir(dir_log):
        dir_log += f'{file_name}_{len(os.listdir(dir_log))}.log'
        with open(dir_log, 'w'):
             os.utime(dir_log, None)
    if not os.path.exists(dir_log):
        os.makedirs(dir_log)
        dir_log += f'{file_name}_{len(os.listdir(dir_log))}.log'
        with open(dir_log, 'w'):
             os.utime(dir_log, None)
    return dir_log


# load data from csv file
def load_csv(file_name):
    #    # Input
    # file_name: file name of mention vectors
    # Output:
    # data: mention vectors
    data = []
    with open(file_name, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        for line in reader:
            vec = [float(l) for l in line]
            data.append(vec)
    return data


# load mention vectors
def load_vectors(mv_path, words=None):
    # mv_path: mention vector path
    # words: word list or None
    logging.info("loading vectors.")
    men_vec = []
    word_list = []
    if words is not None:
        for w in words:
            if w + ".pt" in os.listdir(mv_path):
                vec = torch.load(os.path.join(mv_path, w + ".pt")).tolist()
            elif w + ".csv" in os.listdir(mv_path):
                vec = load_csv(os.path.join(mv_path, w + ".csv"))
            men_vec.extend(vec)
            word_list.extend([w] * len(vec))
            del vec
    else:
        for file in sorted(os.listdir(mv_path)):
            w = file.split('.')[0]
            vec = load_csv(os.path.join(mv_path, file))
            men_vec.extend(vec)
            word_list.extend([w] * len(vec))
            del vec
    logging.info("Total number of vectors is " + str(len(word_list)))
    return np.array(men_vec, dtype="float16"), word_list


def uniq_word_ordered(word_list):
    ## find unique words in given word list, and keep the same order as it
    # Input
    # word_list: e.g. [a, a, a, b, b, b]
    # Output
    # uniq_words: e.g. [a, b]
    uniq_words = list(set(word_list))
    uniq_words.sort(key=word_list.index)
    return uniq_words


# compute knn mv for each mv
def mv_knn(men_vec, word_list, k, out_file, chunk_size=10000, use_cosine=False):
    # Input
    #   men_vec: mention vectors
    #   word_list: words corresponding to mention vectors
    #   k: number of near neighbors
    #   out_file: output mv knn file name
    #   chunk_size: if men_vec is large, compute knn chunk by chunk, to avoid OOM error
    #   use_cosine: if True, use cosine similarity to compute distance between vectors, otherwise L2 distance is used.

    if not os.path.exists(os.path.dirname(out_file)):
        os.makedirs(os.path.dirname(out_file))
    # learn knn vectors for mv
    if not os.path.exists(out_file):
        max_k = 20
        logging.info("Total number of vector is " + str(len(men_vec)))

        men_vec = np.array(men_vec).astype('float32')
        dim = men_vec.shape[1]
        if use_cosine:
            normalize_L2(men_vec)
            index = IndexFlatIP(dim)
        else:
            index = IndexFlatL2(dim)
        # index = faiss.index_cpu_to_all_gpus(index)
        index.add(men_vec)

        logging.info("Total number of index is " + str(index.ntotal))

        knn_idx = []
        s = 0
        times = men_vec.shape[0] // chunk_size + 1
        for tt in range(times):
            if tt == times - 1:
                e = s + men_vec.shape[0] % chunk_size
            else:
                e = s + chunk_size
            _, I = index.search(men_vec[s:e], max_k + 1)
            knn_idx.extend(I)

            s = e
            logging.info(str((tt + 1) * chunk_size / men_vec.shape[0]) + " % vectors processed.")

        np.save(out_file, {"knn": knn_idx, "words": word_list})
        logging.info("knn indices were wrote out. ")

    else:
        data = np.load(out_file, allow_pickle=True).item()
        knn_idx = data['knn']
        word_list = data['words']
    return np.array(knn_idx)[:, 1:k+1], word_list


# load pre-calculated top words of mention vectors
def load_mv_top_words(file_top_words):
    logging.info("Top words pre-computed. Loading it.")
    with contextlib.closing(bz2.BZ2File(file_top_words, "rb")) as f:
        data = json.load(f)
    return data


def dataloader(mv_idx_pairs, batch_size, is_train=False, labels=None):
    mv_idx_pairs = np.array(mv_idx_pairs)
    mv_idx_pairs_tensor = torch.tensor(mv_idx_pairs, dtype=torch.long)
    dataset = TensorDataset(mv_idx_pairs_tensor)
    if is_train:
        if labels is not None:
            sampler = MPerClassSampler(labels, m=1, batch_size=batch_size, length_before_new_iter=len(labels))
            loader = DataLoader(dataset=dataset, batch_size=batch_size, sampler=sampler)
        else:
            loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    else:
        loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)
    return loader


# precision, recall and f1 metric
def pre_rec_f1(y_true, y_pred):
    # y_true: true labels
    # y_pred: predicted labels
    return round(precision_score(y_true, y_pred), 4), round(recall_score(y_true, y_pred), 4), round(f1_score(y_true, y_pred), 4)


# f1 given threshold
def f1(th, y_true, y_score):
    # th: threshold
    # y_true: true labels
    # y_pred: predicted labels
    y_pred = (y_score >= th) * 1
    return -f1_score(y_true, y_pred)


# find the best threshold for classification
def optimal_threshold(y_true, y_score):
    # y_true: true labels
    # y_pred: predicted labels
    bounds = [(np.min(y_score), np.max(y_score))]
    result = brute(f1, args=(y_true, y_score), ranges=bounds, full_output=True, Ns=200)
    return result[0][0], -f1(result[0][0], y_true, y_score)


# preprocess transformed vectors, to fit the input format of cnn classifier
def preprocess_transformed_vectors_for_cnn(transformed_vectors, word2vector, vec_num_per_word=500):
    # transformed_vectors: learned new vectors
    # word2vector: words corresponding to transformed_vectors
    # vec_num_per_word: number of vectors per word
    # output: tensor with shape  word_num * vec_num_per_word * dim
    uniq_word = uniq_word_ordered(word2vector)
    dim = transformed_vectors.shape[1]
    embeddings = torch.zeros((len(uniq_word), vec_num_per_word, dim))

    for i in range(len(uniq_word)):
        wi = uniq_word[i]  # ith word
        where_wi = np.array(word2vector) == wi
        if where_wi.sum() == vec_num_per_word:
            embeddings[i] = transformed_vectors[where_wi]
        else:
            embeddings[i] = torch.cat((transformed_vectors[where_wi],
                                             torch.zeros((vec_num_per_word - where_wi.sum(), dim))), dim=0)
    return embeddings


# preprocess transformed vectors, to fit the input format of svm classifier
def preprocess_transformed_vectors_for_svm(transformed_vectors, word2vector):
    # transformed_vectors: learned new vectors
    # word2vector: words corresponding to transformed_vectors
    # output: tensor with shape  word_num * dim
    uniq_word = uniq_word_ordered(word2vector)
    embeddings = dict()

    for wi in uniq_word:
        where_wi = np.array(word2vector) == wi
        embeddings[wi] = torch.mean(transformed_vectors[where_wi], dim=0).detach().numpy()
    return embeddings


# load pretrained embeddings, e.g. glove, word2vec
def load_baseline_emb(filename):
    embeddings = dict()
    if filename.endswith(".txt"):
        with open(filename, 'r', encoding="utf-8") as f:
            for line in f:
                items = line.strip().split(' ')
                if len(items) < 100:
                    continue
                embeddings[items[0].lower().replace(' ', '_')] = [float(i) for i in items[1:]]
        embeddings['missing_word'] = np.mean(np.array([i for i in embeddings.values()]), axis=0)
    elif filename.endswith(".gz"):
        embeddings = KeyedVectors.load_word2vec_format(filename, binary=True)
        embeddings['missing_word'] = np.mean(embeddings.vectors, axis=0)
    elif filename.endswith(".pickle"):
        with open(filename, 'rb') as f:
            embeddings = pickle.load(f)
        for w in embeddings:
            embeddings[w] = embeddings[w].numpy()
        embeddings['missing_word'] = np.mean(np.array([i for i in embeddings.values()]), axis=0)
    return embeddings


def jaccard_similarity(arr_a, arr_b):
    return len(set(arr_a).intersection(set(arr_b))) / len(set(arr_a).union(set(arr_b)))


def pi_metric(arr_a, arr_b):
    '''
    compute π(x, y)
    :param arr_a: array a, words corresponded to vectors in neigh(x)
    :param arr_b: array b, words corresponded to vectors in neigh(y)
    :return: π(x, y)
    '''
    set_a = set(arr_a)
    set_b = set(arr_b)
    if len(set_a & set_b) > 0:  # if intersection of a and b is not empty
        ab_union = list(set_a.union(set_b))  # all words in set a and b
        cnt_a = [list(arr_a).count(i) if i in arr_a else 0 for i in ab_union]  # count for each word in array a
        cnt_b = [list(arr_b).count(i) if i in arr_b else 0 for i in ab_union]  # count for each word in array a
        sum_of_min = sum(np.min(np.array([cnt_a, cnt_b]), axis=0))
        sum_of_max = sum(np.max(np.array([cnt_a, cnt_b]), axis=0))
        return sum_of_min / sum_of_max
    else:
        return 0


def ispositive(arr_a, arr_b, th, metric='pi'):
    '''
    judge whether a single pair of (index_1, index_2) can be positive pair.
    :param metric: "pi" -> π(x, y) or "jaccard" -> jaccard similarity
    :param arr_a: array a, words corresponded to vectors in neigh(v1)
    :param arr_b: array b, words corresponded to vectors in neigh(v2)
    :param th: threshold
    :return: True or False
    '''
    if metric == "jaccard":
        score = jaccard_similarity(arr_a, arr_b)
    else:
        score = pi_metric(arr_a, arr_b)
    if score > th:
        return True
    else:
        return False


# build positive pairs
def build_positive_pair(men_vec, word_list, method=1, k=5, mv_knn_file="mv_knn.npy", use_cosine=False,
                        word_knn_file="word_knn.json.bz2", olp_th=0.5, out_path='pos_pair'):
    # Input
    #   men_vec: mention vectors
    #   word_list: words corresponding to mention vectors
    #   method: method to build positive pairs.
    #           1 -> if two mvs in the mention vector space are similar, then they are similar
    #           2 -> if two mv's neighbors of word are similar, then they are similar
    #           3 -> if π(x, y) > th, x, y are similar

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    if method == 1:
        out_file = os.path.join(out_path, str(method) + '_' + str(k) + '.npy')
    else:
        out_file = os.path.join(out_path, str(method) + '_' + str(k) + '_' + str(olp_th) + '.npy')

    if os.path.exists(out_file):
        mv_idx_pairs = np.load(out_file)
    else:
        mv_idx_pairs = []
        if method == 1:  # if two in the mention vector space are similar, then they are similar
            knn_idx = mv_knn(men_vec, word_list, k, mv_knn_file, use_cosine=use_cosine)[0]
            # if two mv are neighbors, then they are similar
            mv_idx_pairs = list(zip(np.repeat(range(len(knn_idx)), k), knn_idx.reshape(-1)))

        elif method == 2:  # if two mv's neighbors of word are similar, then they are similar
            word_knn_tmp = np.array(load_mv_top_words(word_knn_file)['top_words'])[:, :k]
            word_knn = []
            for w in uniq_word_ordered(word_list):
                wi = np.where(np.array(word_list) == w)[0]
                word_knn.extend(list(word_knn_tmp[wi]))
            word_knn = np.array(word_knn)

            cnt = 0
            for i, j in list(combinations(range(len(word_knn)), 2)):
                if cnt % 10000 == 0:
                    logging.info(str(cnt) + " pairs processed")
                if ispositive(word_knn[i], word_knn[j], olp_th, metric="jaccard"):
                    mv_idx_pairs.append([i, j])
                cnt += 1
            del word_knn_tmp, word_knn

        elif method == 3:
            knn_idx = mv_knn(men_vec, word_list, k, mv_knn_file, use_cosine=use_cosine)[0]
            word_knn = np.array(word_list)[knn_idx]
            del knn_idx

            cnt = 0
            for i, j in combinations(range(len(word_knn)), 2):
                if cnt % 10000 == 0:
                    logging.info(str(cnt) + " pairs processed")
                if ispositive(word_knn[i], word_knn[j], olp_th, metric="pi"):
                    mv_idx_pairs.append([i, j])
                cnt += 1
            del word_knn

        np.save(out_file, mv_idx_pairs)
    return mv_idx_pairs


def load_text(file_name):
    # load text
    # input: file name of text file
    # return: a list of text, one line per item
    text = []
    with open(file_name, 'r', encoding='utf-8') as f:
        for line in f:
            text.append(line.strip())
    return text


def build_positive_triple(properties, sent_path, out_path):
    '''
    load sentences from sent_path for each property,
    and build positive triples (prop, prompt1, prompt2)
    :param properties: property list from conceptnet
    :param sent_path: sentence path of properties, each property has one sentence file
    :return:
    prompt_pairs: sentence pair list as (prompt1, prompt2)
    prop_prompt_idx: indices of property-prompt_pair
    properties_with_prompts: property list that do have sentences mentioning them
    '''

    file_pp = os.path.join(out_path, "prompt_pairs.txt")
    file_ppi = os.path.join(out_path, "prop_prompt_idx.npy")
    file_prop = os.path.join(out_path, "properties_with_prompts.txt")

    if not os.path.exists(file_pp):
        prompt_pairs = []
        prop_prompt_idx = []
        prompt_pair_idx = 0
        properties_with_prompts = []
        for prop_i in range(len(properties)):
            prop = properties[prop_i]
            if prop + '.txt' not in os.listdir(sent_path):
                continue
            prop_file = os.path.join(sent_path, prop + '.txt')
            prop_sents = load_text(prop_file)   # sentences mentioned a same property, can be positive pair
            prop_sents = uniq_word_ordered(prop_sents)
            prop_sent_pairs = list(combinations(prop_sents, 2))   # positive pair of prop
            prompt_pairs.extend(prop_sent_pairs)
            prop_prompt_idx.extend(list(zip([prop_i] * len(prop_sent_pairs), range(prompt_pair_idx, prompt_pair_idx + len(prop_sent_pairs)))))
            prompt_pair_idx += len(prop_sent_pairs)
            properties_with_prompts.append(prop)

        if not os.path.exists(out_path):
            os.makedirs(out_path)

        np.save(file_ppi, prop_prompt_idx)

        with open(file_pp, 'w', encoding='utf-8') as f:
            f.write('\n'.join(['\t'.join(pp) for pp in prompt_pairs]))

        with open(file_prop, 'w', encoding='utf-8') as f:
            f.write('\n'.join(properties_with_prompts))

    else:
        prop_prompt_idx = np.load(file_ppi)

        prompt_pairs = []
        with open(file_pp, 'r', encoding='utf-8') as f:
            for line in f:
                prompt_pairs.append(line.strip().split('\t'))

        properties_with_prompts = load_text(file_prop)
    return prompt_pairs, prop_prompt_idx, properties_with_prompts


def load_prop_instances(file):
    '''
    load properties and their instances
    :param file: file name of properties and instances
    :return:
    feature_concept: a dictionary, key is property, value is its instances
    words: set of related concepts
    '''
    feature_concept = {}
    words = []
    with open(file, 'r', encoding='utf-8') as f:
        for line in f:
            f, cns = line.strip().split('\t')
            feature_concept[f] = []
            words.extend(cns.split(', '))
            for c in cns.split(', '):
                feature_concept[f].append(c.lower())
    return feature_concept, set(words)


def load_neighbors(file_name, max_k):
    '''
    load mention vectors' knn words
    :param file_name: knn file name
    :param max_k: max value of k
    :return:
    knn: mv's knn words
    '''
    knn = []
    with open(file_name, 'r', encoding='utf-8') as f:
        for line in f:
            noun, nn = line.strip().split('\t')
            knn.append([noun] + nn.split(',')[:max_k])
    return np.array(knn)


def filter_strategy_rosv(knn):
    '''
    filtering strategy used in ijcai paper
    :param knn: mv's knn words
    :return:
    remain_noun_id: remained word index for each mv after filtering
    '''
    nouns = knn[:, 0]
    noun_start_id = {}
    old_noun = ''
    for i in range(nouns.shape[0]):
        cur_noun = nouns[i]
        if old_noun == '' or not cur_noun == old_noun:
            noun_start_id[cur_noun] = i
            old_noun = cur_noun

    k = knn.shape[1] - 1
    target = nouns.repeat(k).reshape((-1, k))
    score = (knn[:, 1:] == target).astype(np.int).sum(axis=1)
    remain_idx = np.where(score < k)[0]
    remain_noun_id = {}
    for id in remain_idx:
        noun = nouns[id]
        if noun not in remain_noun_id:
            remain_noun_id[noun] = [id-noun_start_id[noun]]
        else:
            remain_noun_id[noun].append(id-noun_start_id[noun])
    return remain_noun_id


def init_word_embedding_for_cnn(embeddings, all_words, target_words, remain_word_id=None):
    _, vn, dim = embeddings.size()
    result = torch.zeros((len(target_words), vn, dim))
    for w in target_words:
        if w in all_words:
            wi_emb = all_words.index(w)
        if w in target_words:
            wi_tar = target_words.index(w)
        if remain_word_id is not None and w in remain_word_id:
            emb = embeddings[wi_emb][remain_word_id[w]]
            result[wi_tar] = torch.cat((emb, torch.zeros((vn - emb.size()[0], dim))))
        else:
            result[wi_tar] = embeddings[wi_emb]
    return result


def process_filtered_data_for_cnn(embeddings, all_words, pos_words, neg_words, remain_word_id=None):
    pos_data = init_word_embedding_for_cnn(embeddings, all_words, pos_words, remain_word_id)
    neg_data = init_word_embedding_for_cnn(embeddings, all_words, neg_words, remain_word_id)

    data = torch.cat((pos_data, neg_data), dim=0)
    label = np.array(
            [1] * pos_data.shape[0] + [0] * neg_data.shape[0])
    return data, label


def preprocess_filter_transformed_vectors_for_svm(transformed_vectors, word2vector, remain_word_id):
    uniq_word = uniq_word_ordered(word2vector)
    embeddings = dict()

    for w in uniq_word:
        where_w = np.array(word2vector) == w
        if w in remain_word_id:
            embeddings[w] = torch.mean(transformed_vectors[where_w][remain_word_id[w]], dim=0)
        else:
            embeddings[w] = torch.mean(transformed_vectors[where_w], dim=0)
    return embeddings


def write_csv(file, properties, results):
    '''
    write results (map, precision, recall, f1) of each property into a csv file
    :param file: output file name
    :param properties: property list
    :param results: map, precision, recall, f1 of properties
    '''
    with open(file, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['property', 'MAP', 'pre', 'rec', 'f1'])
        for i in range(len(properties)):
            writer.writerow([properties[i]] + results[i])
        writer.writerow(['mean'] + np.round(np.mean(np.array(results), axis=0), decimals=4).tolist())


def write_txt(file, results, header):
    '''
    write avgerage results of all properties into a text file
    :param file: output file name
    :param results: map, precision, recall, f1 of properties
    :param header: string mentioning used parameters
    '''
    output = header
    output += 'MAP\tpre\trec\tf1\n'
    print(results)
    output += '\t'.join([str(m) for m in np.round(np.mean(np.array(results), axis=0), decimals=4)]) + '\n'
    output += '\n\n'
    with open(file, 'a+', encoding='utf-8') as fout:
        fout.writelines(output)


def get_word_list(mv_knn_file):
    '''
    get word list corresponding to mention vectors
    :param mv_knn_file: file name of mv knn
    :return: word list corresponding to mention vectors
    '''
    return np.load(mv_knn_file, allow_pickle=True).item()['words']


def init_word_embedding_for_svm(embeddings, word_list):
    '''
    get embeddings of words in word list, used for svm
    :param embeddings: a dictionary, key is word, value is its embedding
    :param word_list: words that need embeddings
    :return:
    words_embeddings: embeddings of words in word list
    '''
    words_embeddings = []
    for ww in word_list:
        if ww in embeddings:
            words_embeddings.append(embeddings[ww])
        elif "missing_word" in embeddings:
            # print(ww)
            words_embeddings.append(embeddings['missing_word'])
    return words_embeddings


def process_data_for_svm(embeddings, pos_words, neg_words):
    '''
    get embedding for positive and negative words for svm
    :param embeddings: a dictionary, key is word, value is its embedding
    :param pos_data: positive words
    :param neg_data: negative words
    :return:
    '''
    pos_data = init_word_embedding_for_svm(embeddings, pos_words)
    neg_data = init_word_embedding_for_svm(embeddings, neg_words)

    labels = np.array([1] * len(pos_data) + [0] * len(neg_data))
    data = np.array(pos_data + neg_data)
    return data, labels