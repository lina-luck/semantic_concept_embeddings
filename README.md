# Semantic_concept_embeddings
The code and datasets of "Distilling Semantic Concept Embeddings from Contrastively Fine-Tuned Language Models" presented in SIGIR'23.

# Requirements
- Python 3.7
- transformers == 4.18.0
- scikit-learn == 1.0.2
- faiss == 1.7.2
- pytorch-metric-learning == 1.0.0
- gensim == 4.1.2

# Usage
Take the WordNet dataset as an example to show how to run the models.
### Preprocessing
Firstly, collect sentences that mention the concept, one file for each concept. Store these sentence files in path ./sents_64_500.  <br><br>
Secondly, extract masked mention vectors for concepts and store mention vectors to the path ./bert_mask_mv/
```
python ./src/extract_mask_mv.py -w_file ./dataset/wordnet/wordnet.txt -batch_size 32 -max_seq_len 128 -bert_version bert-base-cased -out_path ./bert_mask_mv/ -sent_path ./sents_64_500
```
Thirdly, calculate the top k nearest concepts for each concept, and store them, used for constructing weakly labeled examples by neighborhood structure.
```
python ./src/caculate_topwords.py -w_file ./dataset/wordnet/wordnet.txt -mv_path ./bert_mask_mv/ -k 5 -top_word_path ./dataset/wordnet/ -use_cosine
```
Finally, extract sentences mentioning both concept and property for constructing weakly labeled examples by ConceptNet
```
python ./src/extract_sentences.py -wiki_file file_of_wikipedia_sentences -cp_file ./dataset/ds_data/concept_property.txt -out_path ./dataset/ds_data/prop_sentences
```
### Run ConProj
```
python ./src/run_cl.py -w_file ./dataset/wordnet/wordnet.txt -dataset wordnet -method 3 -batch_size 256 -in_dim 1024 -k 10 -lr 2e-5 -use_hard_pair -tau 0.05 -lr_schedule cosine -hidden_dim 512 -out_dim 256 -olp_th 0.7 -use_cosine -out_path cl_emb -mv_knn_file ./mv_knn/wordnet/cos_20_500.npy -word_knn_file ./dataset/wordnet/closest5_top_words_cos.json.bz2 -mv_path ./bert_mask_mv/
```
### Run ConFT
```
python ./src/run_finetune.py -w_file ./dataset/wordnet/wordnet.txt -max_seq_len 64 -bert_version ./pretrained_models/bert-large-uncased -lr 2e-5 -use_hard_pair -lr_schedule cosine -loss infonce -hidden_dim 512 -out_dim 256 -tau 0.05 -batch_size 256 -sent_path ./sents_64_500 -mv_knn_file ./mv_knn/wordnet/cos_20_500.npy -dataset wordnet -method 3 -k 10 -use_cosine -olp_th 0.7  -out_path ./finetune_bert_emb/ -word_knn_file ./dataset/wordnet/closest5_top_words_cos.json.bz2 -mv_path ./bert_mask_mv/
```
### Run ConCN
Firstly, finetuning the BERT model using distant supervision information from ConceptNet
```
python ./src/run_finetune_with_ds.py -prop_file ./dataset/ds_data/properties.txt -prop_num 5000 -ppt_path ./dataset/ds_data/prop_sentences -bert_version ./pretrained_models/bert-large-uncased -batch_size 256 -lr 2e-5 -use_hard_pair -tau 0.05 -lr_schedule cosine -loss infonce -hidden_dim 512 -out_dim 256
```
Secondly, extracting mention vectors from fine-tuned BERT model
```
python ./src/extract_mv.py -w_file ./dataset/wordnet/wordnet.txt -bert_version ./pretrained_models/bert-large-uncased -model ./finetuned_model_with_ds/bert-large-uncased_infonce_0.05_2e-05_256_0.001.pt -loss infonce -use_hard_pair -tau 0.05 -out_path mv_ds_bert_large -sent_path ./sents_64_500 -batch_size 100 -max_seq_len 64
```
### Run classifier
SVM as classifier
```
python ./src/run_classifier.py -data_dir ./dataset -dataset wordnet -mv_knn_file ./mv_knn/wordnet/cos_20_500.npy -emb_path ./mv_ds_bert_large/bert-large-uncased_infonce_0.05_2e-05_256_0.001 -res_path result -cls svm
```
CNN as classifier
```
python ./src/run_classifier.py -data_dir ./data -dataset wordnet -mv_knn_file ./mv_knn/wordnet/cos_20_500.npy -emb_path ./mv_ds_bert_large/bert-large-uncased_infonce_0.05_2e-05_256_0.001 -res_path result -vec_num_per_word 500 -cls cnn

```
# Citation
```
@inproceedings{Li_conceptembddings_sigir23,
author = {Li, Na and Kteich, Hanane and Bouraoui, Zied and Schockaert, Steven},
title = {Distilling Semantic Concept Embeddings from Contrastively Fine-Tuned Language Models},
year = {2023},
isbn = {9781450394086},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3539618.3591667},
doi = {10.1145/3539618.3591667},
booktitle = {Proceedings of the 46th International ACM SIGIR Conference on Research and Development in Information Retrieval},
pages = {216–226},
numpages = {11},
keywords = {language models, commonsense knowledge, contrastive learning, word embedding},
location = {Taipei, Taiwan},
series = {SIGIR '23}
}
```
