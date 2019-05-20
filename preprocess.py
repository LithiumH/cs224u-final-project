import os
import subprocess
import json
import pickle
import numpy as np
from multiprocessing import Pool
from shutil import rmtree

CNN_BASE = os.path.join('data', 'cnn')
DM_BASE = os.path.join('data', 'dailymail')

CNN_STORY_DIR = os.path.join(CNN_BASE, 'stories')
DM_STORY_DIR = os.path.join(DM_BASE, 'stories')

CNN_STORY_TOKENIZED = os.path.join('data', 'cnn', 'stories-tokenized')
DM_STORY_TOKENIZED = os.path.join('data', 'dailymail', 'stories-tokenized')

SRC_PK = os.path.join('data', 'src.pk')
TGT_PK = os.path.join('data', 'tgt.pk')
dm_single_close_quote = u'\u2019' # unicode
dm_double_close_quote = u'\u201d'
END_TOKENS = ['.', '!', '?', '...', "'", "`", '"', dm_single_close_quote, dm_double_close_quote, ")"] # acceptable ways to end a sentence
SENTENCE_START = '<s>'
SENTENCE_END = '</s>'

def tokenize_stories(stories_dir, tokenized_stories_dir):

    print("Preparing to tokenize %s to %s..." % (stories_dir, tokenized_stories_dir))
    stories = os.listdir(stories_dir)
    # make IO list file
    print("Making list of files to tokenize...")
    with open("mapping_for_corenlp.txt", "w") as f:
        for s in stories:
            if (not s.endswith('story')):
                continue
            f.write("%s\n" % (os.path.join(stories_dir, s)))
    command = ['java', 'edu.stanford.nlp.pipeline.StanfordCoreNLP' ,'-annotators', 'tokenize,ssplit', '-ssplit.newlineIsSentenceBreak', 'always', '-filelist', 'mapping_for_corenlp.txt', '-outputFormat', 'json', '-outputDirectory', tokenized_stories_dir]
    print("Tokenizing %i files in %s and saving in %s..." % (len(stories), stories_dir, tokenized_stories_dir))
    subprocess.call(command)
    print("Stanford CoreNLP Tokenizer has finished.")
    os.remove("mapping_for_corenlp.txt")

    # Check that the tokenized stories directory contains the same number of files as the original directory
    num_orig = len(os.listdir(stories_dir))
    num_tokenized = len(os.listdir(tokenized_stories_dir))
    if num_orig != num_tokenized:
        raise Exception(
            "The tokenized stories directory %s contains %i files, but it should contain the same number as %s (which has %i files). Was there an error during tokenization?" % (
            tokenized_stories_dir, num_tokenized, stories_dir, num_orig))
    print("Successfully finished tokenizing %s to %s.\n" % (stories_dir, tokenized_stories_dir))

def percentage_in_src_vocab(src, tgt):
    src_vocab = set()
    for sent in src:
        src_vocab |= set(sent)
    count = 0
    total_len = 0
    for sent in tgt:
        for word in sent:
            if word in src_vocab:
                count += 1
            total_len += 1
    return count / total_len

def process_json(filename):
    src, tgt = [], [] # a document is a list of list of words
    highlight = False # highlights are always at the end of the document
    f = open(filename, 'r')
    parsed = json.load(f)
    for sent in parsed['sentences']:
        words = [word['word'] for word in sent['tokens']]
        if words[-1] not in END_TOKENS:
            words += ['.']
        if words[0] == '@highlight':
            highlight = True
        elif highlight:
            tgt += [words]
        else:
            src += [words]
    return src, tgt

def process_all_json(file_dir):
    pool = Pool(processes=10)
    srcs, tgts = [], []
    percentages = []
    file_paths = [os.path.join(file_dir, file_name) for file_name in os.listdir(file_dir)]
    for tup in pool.imap_unordered(process_json, file_paths):
        src, tgt = tup
        srcs.append(src)
        tgts.append(tgt)
        percentages.append(percentage_in_src_vocab(src, tgt))
    print("percentage of summary vocab in source is {}".format(np.mean(percentages)))
    return srcs, tgts


if __name__ == '__main__':
    print("Tokenizing stories with Stanford CoreNLP and saving to {} and {}".format(CNN_STORY_TOKENIZED, DM_STORY_TOKENIZED))
    # tokenize_stories(CNN_STORY_DIR, CNN_STORY_TOKENIZED)
    # tokenize_stories(DM_STORY_DIR, DM_STORY_TOKENIZED)

    print("processing tokenized stories into src and tgt pickle files")
    srcs_cnn, tgts_cnn = process_all_json(CNN_STORY_TOKENIZED)
    srcs_dm, tgts_dm = process_all_json(DM_STORY_TOKENIZED)
    src, tgt = srcs_cnn + srcs_dm, tgts_cnn + tgts_dm
    for f_name, obj in zip([SRC_PK, TGT_PK], [src, tgt]):
        with open(f_name, 'wb') as f:
            pickle.dump(obj, f)
    print("removing entire directory")
    rmtree(CNN_BASE)
    rmtree(DM_BASE)


