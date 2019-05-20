import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

import os
import subprocess
import json
import pickle
from multiprocessing import Pool

import numpy as np
import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel

from sklearn.model_selection import train_test_split

from rouge import Rouge

SRC_PK = os.path.join('data', 'src.pk')
TGT_PK = os.path.join('data', 'tgt.pk')
GLOVE_HOME = os.path.join('data', 'glove.840B.300d.txt')
PROCESSED_PK = os.path.join('data', 'processed.pk')

DOCS = os.path.join('data', 'docs.pk')
TAGS = os.path.join('data', 'tags.pk')
TAGGED_SUMS = os.path.join('data', 'tagged_sums.pk')
GOLD_SUMS = os.path.join('data', 'gold_sums.pk')
IDX_TAGS = os.path.join('data', 'idx_tags.pk')
IDS = os.path.join('data', 'idx.pk')

print('premain: load BertTokenizer')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', max_len=510)

def tag(doc, tgt):
    """
    doc: a list of src tokens
    tgt: a list of tgt tokens that we will look for in the doc
    """
    if len(tgt) == 0:
        print('zero sized tgt')
        return None
    vocab = set(tgt)
    doc = np.array(doc)
    tgt = np.array(tgt)

    label = np.zeros(len(doc), dtype=bool)
    ## The following tags all tokens present in both the source and target
#     for i in range(len(doc)):
#         if doc[i] in vocab:
#             label[i] = 1
    ## The following does the max tagging thingy the original paper did
    l, r = 0, 0
    while r < len(tgt):
        old_idxs = []
        idxs = [(i,i+1) for i, token in enumerate(doc) if token == tgt[r]]
        while len(idxs) > 0 and r + 1 < len(tgt):
            r += 1
            old_idxs, idxs = idxs, []
            for idx in old_idxs:
                if idx[-1] < len(doc) and doc[idx[-1]] == tgt[r]:
                    idxs.append((idx[0], idx[-1] + 1))
        if len(idxs) > 0: ## we ran out of tgt
            label[idxs[0][0]:idxs[0][-1]] = 1
            break
        elif len(old_idxs) > 0: ## we found longest seq
            label[old_idxs[0][0]:old_idxs[0][-1]] = 1
        else: ## this token does not exist
            r += 1
    idxs = []
    for i in range(len(tgt)):
        idxs.append(list(np.argwhere(doc == tgt[i]).flatten()))
    return label, idxs

def process_src_tgt(srcs, tgts, start_idx=0, end_idx=-1):
    assert len(srcs) == len(tgts)
    docs, tags = [], []
    tagged_sum, gold_sum_bert, gold_sum_idxs = [], [], []
    ranges = []
    rn = range(len(srcs)) if end_idx == -1 else range(start_idx, end_idx)
    for i in rn:
        ## process src
        sents = [' '.join(sent) + ' [SEP]' for sent in srcs[i]]
        doc = ' '.join(['[CLS]'] + sents)
        doc = tokenizer.tokenize(doc)[:510]

        ## process tgt
        tgt = ' '.join([' '.join(sent) for sent in tgts[i]])
        tgt = tokenizer.tokenize(tgt)[:110]
        label, idxs = tag(doc, tgt)

        ## generate tagged_summary for oracle rouge
        tagged = []
        for idx in idxs:
            doc = np.array(doc)
            if len(doc[idx]) > 0:
                tagged.append(doc[idx][0])

        ## Add both to list
        docs.append(tokenizer.convert_tokens_to_ids(doc))
        tags.append(label)
        tagged_sum.append((' '.join(tagged)).replace(' ##', ''))
        gold_sum_bert.append(' '.join(tgt).replace(' ##', ''))
        gold_sum_idxs.append(idxs)
        ranges.append(i)
    return docs, tags, tagged_sum, gold_sum_bert, np.array(gold_sum_idxs), ranges

def check_strictly_increasing(nested_sequence):
    counter = 0
    for sequence in nested_sequence:
        for i in sequence:
            if i != counter:
                return False
            counter += 1
    return True

def clean(lst, valid_ids):
    return [lst[i] for i in valid_ids]

def process_ranges(args):
    return process_src_tgt(src, tgt, args[0], args[1])

if __name__ == '__main__':
    print("loading src and tgt")
    f = open(SRC_PK, 'rb')
    src = pickle.load(f)
    f.close()

    f = open(TGT_PK, 'rb')
    tgt = pickle.load(f)
    f.close()

    print("processing sequences")
    n = 20
    pool = Pool(n)
    k = len(src)//n
    # k = 350//n
    result = pool.map(process_ranges, [(start * k, (start+1) * k) for start in range(n)])
    strictly_increasing = check_strictly_increasing([tup[-1] for tup in result])
    print("ranges in the result is strictly increasing? {}".format(strictly_increasing))

    src, tgt = None, None
    docs, tags, tagged_sums, gold_sums_bert, gold_sums_idxs, ids = [], [], [], [], [], []
    for a, b, c, d, e, f in result:
        valid_ids = [i for i in range(len(a)) if len(c[i]) > 0 and len(d[i]) > 0]
        docs.extend(clean(a, valid_ids))
        tags.extend(clean(b, valid_ids))
        tagged_sums.extend(clean(c, valid_ids))
        gold_sums_bert.extend(clean(d, valid_ids))
        gold_sums_idxs.extend(clean(e, valid_ids))
        ids.extend(clean(f, valid_ids))
    print("checkpointing into many files")
    for obj, fname in zip([docs, tags, tagged_sums, gold_sums_bert, gold_sums_idxs, ids],
                          [DOCS, TAGS, TAGGED_SUMS, GOLD_SUMS, IDX_TAGS, IDS]):
        with open(fname, 'wb') as f:
            pickle.dump(obj, f)

    print("calculating rogue score of tagged summaries (tokenized by Bert)")
    rouge = Rouge()
    scores = rouge.get_scores(tagged_sums, gold_sums_bert, avg=True)
    print("Roge scores: {}".format(scores))
    processed = dict()
    processed['rogue_oracle'] = scores

    print("Splitting data into train/dev/test/tiny")
    X_train, X_dev_test, y_tags_train, y_tags_dev_test, y_decode_train, y_decode_dev_test, ids_train, ids_dev_test = \
            train_test_split(docs, tags, gold_sums_idxs, ids, test_size=0.1)
    X_dev, X_test, y_tags_dev, y_tags_test, y_decode_dev, y_decode_test, ids_dev, ids_test =\
            train_test_split(X_dev_test, y_tags_dev_test, y_decode_dev_test, ids_dev_test, test_size=0.5)
    X_tiny, y_tags_tiny, y_decode_tiny, ids_tiny = \
            X_train[:5000], y_tags_train[:5000], y_decode_train[:5000], ids_train[:5000]
    processed = dict()
    processed['train'] = {'X':X_train, 'y_tag':y_tags_train, 'y_decode':y_decode_train,
            'ids':ids_train}
    processed['dev'] = {'X':X_dev, 'y_tag':y_tags_dev, 'y_decode':y_decode_dev,
            'ids':ids_dev}
    processed['test'] = {'X':X_test, 'y_tag':y_tags_test, 'y_decode':y_decode_test,
            'ids':ids_test}
    processed['tiny'] = {'X':X_tiny, 'y_tag':y_tags_tiny, 'y_decode':y_decode_tiny,
            'ids':ids_tiny}
    print("checkpointing into a file")
    with open(PROCESSED_PK, 'wb') as f:
        pickle.dump(processed, f)




