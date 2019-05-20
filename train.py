import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

import os
import subprocess
import json
import pickle
import argparse
import logging
from multiprocessing import Pool

import numpy as np
import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel

from sklearn.model_selection import train_test_split

from torch.nn.init import xavier_uniform_
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as sched
import torch.utils.data as data

from tensorboardX import SummaryWriter
from tqdm import tqdm

from rouge import Rouge

from models import SummarizerLinear
import util

PROCESSED_DATA = os.path.join('data', 'data.pk')
with open(PROCESSED_DATA, 'rb') as f:
    all_data = pickle.load(f)

GOLD_SUMS = os.path.join('data', 'gold_sums.pk')
with open(GOLD_SUMS, 'rb') as f:
    gold_sums = pickle.load(f)

class SummarizationDataset(data.Dataset):
    def __init__(self, X, y, ids):
        super(SummarizationDataset, self).__init__()
        assert len(X) == len(y) == len(ids)
        self.X = X
        self.y = y
        self.ids = ids
    def __getitem__(self, i):
        return (self.X[i], self.y[i], self.ids[i])
    def __len__(self):
        return len(self.X)

def collate_fn(examples):
    """
    collate function requires all examples to be non-padded
    """

    def merge_1d(arrays, dtype=torch.int64, pad_value=0):
        lengths = [len(a) for a in arrays]
        padded = torch.zeros(len(arrays), max(lengths), dtype=dtype)
        for i, seq in enumerate(arrays):
            end = lengths[i]
            padded[i, :end] = torch.tensor(seq)[:end]
        return padded
    X, y, ids = zip(*examples)
    return merge_1d(X), merge_1d(y), torch.tensor(ids, dtype=torch.int64)

def train(args):
    args.save_dir = util.get_save_dir(args.save_dir, args.name, training=True)
    log = util.get_logger(args.save_dir, args.name)
    tbx = SummaryWriter(args.save_dir)
#     device, args.gpu_ids = util.get_available_devices()
    device, args.gpu_ids = torch.device('cpu'), []

    # Set random seed
    log.info('Using random seed {}...'.format(args.seed))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    log.info('Building model...')
    model = SummarizerLinear()
#     model = nn.DataParallel(model, args.gpu_ids)
    if args.load_path:
        log.info('Loading checkpoint from {}...'.format(args.load_path))
        model, step = util.load_model(model, args.load_path, args.gpu_ids)
    else:
        step = 0
    model = model.to(device)

    model.train()
    optimizer = optim.Adam(model.parameters(), args.lr,
                               weight_decay=args.l2_wd)

    log.info('Building dataset...')
    train_dataset = SummarizationDataset(all_data['tiny']['X'], all_data['tiny']['y'], all_data['tiny']['ids'])
    dev_dataset = SummarizationDataset(all_data['tiny']['X'], all_data['tiny']['y'], all_data['tiny']['ids'])
    train_loader = data.DataLoader(train_dataset,
                                   batch_size=args.batch_size,
                                   num_workers=args.num_workers,
                                   shuffle=True,
                                   collate_fn=collate_fn)
    dev_loader = data.DataLoader(dev_dataset,
                                   batch_size=args.batch_size,
                                   num_workers=args.num_workers,
                                   shuffle=False,
                                   collate_fn=collate_fn)
    ## Train!
    log.info('Training...')
    steps_till_eval = args.eval_steps
    epoch = step // len(train_dataset)
    while epoch != args.num_epochs:
        epoch += 1
        log.info('Starting epoch {}...'.format(epoch))
        with torch.enable_grad(), \
                tqdm(total=len(train_loader.dataset)) as progress_bar:
            for X, y, _ in train_loader:
                batch_size = X.size(0)
                X = X.to(device)
                optimizer.zero_grad()

                logits, mask = model(X)
                y = y.float().to(device)
                loss = (F.binary_cross_entropy_with_logits(logits, y, reduction='none') * mask).mean()
                loss_val = loss.item()

                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                # scheduler.step(step // batch_size)

                # Log info
                step += args.batch_size
                progress_bar.update(args.batch_size)
                progress_bar.set_postfix(epoch=epoch,
                                         Loss=loss_val)
                tbx.add_scalar('train/Loss', loss_val, step)
                tbx.add_scalar('train/LR',
                               optimizer.param_groups[0]['lr'],
                               step)

                steps_till_eval -= batch_size
                if steps_till_eval <= 0:
                    steps_till_eval = args.eval_steps

                    # Evaluate and save checkpoint
                    log.info('Evaluating at step {}...'.format(step))
                    results, pred_dict = evaluate(model, dev_loader, device)
                    if results is None:
                        log.info('Selected predicted no select for all in batch')
                        continue
                    saver.save(step, model, results[args.metric_name]['r'], device)

#                     # Log to console
                    results_str = ', '.join('{}: {:05.2f}'.format(k, v['r'])
                                            for k, v in results.items())
                    log.info('Dev {}'.format(results_str))

                    # Log to TensorBoard
                    log.info('Visualizing in TensorBoard...')
                    for k, v in results.items():
                        tbx.add_scalar('dev/{}'.format(k), v['r'], step)
#                     util.visualize(tbx,
#                                    pred_dict=pred_dict,
#                                    eval_path=args.dev_eval_file,
#                                    step=step,
#                                    split='dev',
#                                    num_visuals=args.num_visuals)

def evaluate(model, data_loader, device):
    model.eval()
    all_preds = []
    gold_summaries = [] # tokenized gold summaries
    with torch.no_grad(), \
            tqdm(total=len(data_loader.dataset)) as progress_bar:
        for X, y, ids in data_loader:
            # Setup for forward
            batch_size = X.size(0)
            logits, mask = model(X)
            y = y.float().to(device)
            loss = (F.binary_cross_entropy_with_logits(logits, y, reduction='none') * mask).mean()

            # Log info
            progress_bar.update(batch_size)
            progress_bar.set_postfix(Loss=loss.item())

            preds = util.untokenize(X, logits, mask)
            all_preds.extend(preds)
            gold_summaries.extend(np.array(gold_sums)[np.array(ids)])

    model.train()

    valid_ids = [i for i in range(len(all_preds)) if len(all_preds[i]) > 0]
    all_pred = [all_pred[i] for i in valid_ids]
    gold_summaries = [gold_summaries[i] for i in valid_ids]
    if len(valid_ids) == 0:
        return None, None
    rouge = Rouge()
    results = rouge.get_scores(all_preds, gold_sums, avg=True)

    return results, all_preds


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-seed", default=1234)
    parser.add_argument("-load_path", default=None)
    parser.add_argument("-split", default='tiny')
    parser.add_argument("-batch_size", default=8)
    parser.add_argument("-gpu_ids", default=[0])
    parser.add_argument("-num_workers", default=1)
    parser.add_argument("-lr", default=0.001)
    parser.add_argument("-l2_wd", default=0)
    parser.add_argument("-eval_steps", default=1)
    parser.add_argument("-num_epochs", default=1)
    parser.add_argument("-max_grad_norm", default=2)
    parser.add_argument("-save_dir", default='saved_models')
    parser.add_argument("-name", default='default')
    parser.add_argument("-metric_name", default='rouge-1')
    args = parser.parse_args([])

    train(args)
