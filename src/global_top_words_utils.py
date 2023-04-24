## util files for compute top words for each mention vector
import os
import logging
import sys
import bz2, json, contextlib
project_path = os.path.split(os.path.abspath(os.path.realpath(__file__)))[0] + "/../"
sys.path.append(os.path.abspath(project_path))
from src.utils import load_vectors, uniq_word_ordered
import numpy as np
from faiss import normalize_L2, IndexFlatL2, IndexFlatIP
from sklearn.metrics.pairwise import cosine_similarity


def top_k(word_list, k, vectors, use_cosine=False, closest_k=1):
    ## compute k nearest words for all mention vector
    # Input:
    # k: number of neighbors
    # vectors: all mention vectors
    # use_cosine: whether to use cosine similarity to compute distance. if False, use Euclidean distance
    # closest_k: closest k vectors to be averaged when compute the distance from mention vector to a word

    # Output:
    # top_words

    logging.info("Compute top words for each mention vector")
    #global word_list
    vectors = np.array(vectors).astype("float32")
    dim = vectors.shape[1]
    if use_cosine:
        normalize_L2(vectors)
        index = IndexFlatIP(dim)
        max_dis = -np.inf
    else:
        index = IndexFlatL2(dim)
        max_dis = np.inf

    uniq_words = uniq_word_ordered(word_list)

    # choose nearest neighbor from mv of each word for each vector
    vec2word_distances = []  # len = len(uniq_word)
    import sys
    cnt = 0
    logging.info("Memory usage of vectors is " + str(round(sys.getsizeof(vectors) / 1048576, 4)) + " MB")
    for word in uniq_words:
        if cnt % 1000 == 0:
            logging.info(str(cnt) + " of " + str(len(uniq_words)) + " processed")
        cnt += 1
        wi = np.where(np.array(word_list) == word)[0]
        vectors_wi = vectors[wi]
        index.add(vectors_wi)
        Ii = index.search(vectors, closest_k)[1]  # nearest neighbor for each vector
        index.reset()

        mean_vectors_wi = np.mean(vectors_wi[Ii], axis=1)

        if use_cosine:
            di = cosine_similarity(vectors, mean_vectors_wi)
            di = di[np.diag_indices_from(di)]
        else:
            di = np.linalg.norm(vectors - mean_vectors_wi, axis=1)

        di[wi] = max_dis
        vec2word_distances.append(di)
        del di

    # ranking
    if use_cosine:
        word_ranking = [
            sorted(range(len(vec2word_distances)), key=lambda i: [jj[j] for jj in vec2word_distances][i], reverse=True)
            for j in range(len(vec2word_distances[0]))]
        word_ranking = np.array(word_ranking, dtype="uint16").transpose()
    else:
        word_ranking = np.argsort(vec2word_distances, axis=0).astype("uint16")  # close -> far for l2 distance
    top_words = np.array(uniq_words)[word_ranking[:k]].transpose()  # k * len(v)
    del word_ranking
    return top_words


def global_top_words(path, mv_path, k, use_cosine=False, words=None, closest_k=1):
    ## compute top words for each mention vector
    ## Input:
    # path: output path to store the resulted top_words
    # mv_path: input path of initial mention vector files
    # k: number of neighbors
    # use_cosine: whether to use cosine similarity to compute distance. if False, use Euclidean distance
    # words: word list to be handle
    # keep_vectors: whether to keep loaded vectors, if False, remove after using it to release memory space
    # closest_k: closest k vectors to be averaged when compute the distance from mention vector to a word
    # clus_num: cluster number, used for kmeans
    # min_clus_size: used for hdbscan clustering

    ## Output:
    # top_words: top k words for all mention vectors

    # build out path if it does not exist
    if not os.path.exists(path):
        os.makedirs(path)

    if use_cosine:
        file_top_words = os.path.join(path, "top_words_cos.json.bz2")
    else:
        file_top_words = os.path.join(path, "top_words_L2.json.bz2")

    file_top_words = file_top_words.replace("top", "closest" + str(closest_k) + "_top")

    if not os.path.exists(file_top_words):
        logging.info("Compute top words.")
        vectors, word_list = load_vectors(mv_path, words)
        top_words = top_k(word_list, k, vectors, use_cosine, closest_k)
        del vectors
        with contextlib.closing(bz2.BZ2File(file_top_words, "wb")) as f:
            json_str = json.dumps({'top_words': top_words.tolist(), 'word_list': word_list}).encode()
            f.write(json_str)
    else:
        raise FileExistsError("top word file have computed")
