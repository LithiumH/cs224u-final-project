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

from models import SummarizerLinear, SummarizerAbstractive, GreedyDecoder, SummarizerLinearAttended
import util

print("premain: loading all data")
# PROCESSED_DATA = os.path.join('data', 'data.pk')
PROCESSED_DATA = os.path.join('data', 'super_tiny.pk')
with open(PROCESSED_DATA, 'rb') as f:
    all_data = pickle.load(f)

class SummarizationDataset(data.Dataset):
    def __init__(self, X, y, gold_sums):
        super(SummarizationDataset, self).__init__()
        assert len(X) == len(y) == len(gold_sums)
        self.X = X
        self.y = y
        self.gold_sums = gold_sums
    def __getitem__(self, i):
        return (self.X[i], self.y[i], self.gold_sums[i])
    def __len__(self):
        return len(self.X)

def collate_fn(examples):
    """
    collate function requires all examples to be non-padded
    """

    def merge_tag(arrays, dtype=torch.int64, pad_value=0):
        """
        This function is used for tagging task only. We can check the
        shape of the arrays to know when to call this function
        """
        lengths = [len(a) for a in arrays]
        padded = torch.zeros(len(arrays), max(lengths), dtype=dtype)
        for i, seq in enumerate(arrays):
            end = lengths[i]
            padded[i, :end] = torch.tensor(np.array(seq, dtype=int))[:end]
        return padded

    def merge_decode(arrays, src_max_len, dtype=torch.int64, pad_value=0):
        """
        This function is used for decoding task only.
        It creates "masks" from the indecies :
        [1, 3] -> [[0, 1, 0, 0], [0, 0, 0, 1]]
        Note the returned mask is of tgt_length + 1 to account for the offsets of the begin token.
        """
        tgt_max_len = max(len(a) for a in arrays)
        padded = torch.zeros(len(arrays), tgt_max_len + 1, src_max_len, dtype=dtype)
        padded[:, 0, 0] = 1 # CLS is the start of all abstracts
        for i, seq in enumerate(arrays):
            c = 0 # this is used to truncate all the y's that are "empty"
            for lst in seq:
                if len(lst) > 0:
                    padded[i, c + 1].scatter_(0, torch.tensor(lst, dtype=torch.int64), 1)
                    c += 1
        return padded
    merge_X = merge_tag # same merge function

    X, y, gold_sums = zip(*examples)
    X = merge_X(X)
    if type(y[0][0]) == type(np.array([])): # we are doing decoding task
        y = merge_decode(y, max(len(a) for a in X))
    else:
        y = merge_tag(y)
    return X, y, gold_sums

def train(args):
    args.save_dir = util.get_save_dir(args.save_dir, args.name, training=True)
    log = util.get_logger(args.save_dir, args.name)
    tbx = SummaryWriter(args.save_dir)
    if args.gpu_ids == 'cpu':
        device, args.gpu_ids = torch.device('cpu'), []
    else:
        device, args.gpu_ids = util.get_available_devices()
    log.info('training on device {}'.format(str(device)))

    # Set random seed
    log.info('Using random seed {}...'.format(args.seed))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    log.info('Building model...')
    if args.task == 'tag':
        model = SummarizerLinearAttended(128, 256)
#         model = SummarizerLinear()
    else:
        model = SummarizerAbstractive(128, 256, device)

#     model = nn.DataParallel(model, args.gpu_ids)
    if args.load_path:
        log.info('Loading checkpoint from {}...'.format(args.load_path))
        model, step = util.load_model(model, args.load_path, args.gpu_ids)
    else:
        step = 0
    model = model.to(device)

    model.train()

    ## get a saver
    saver = util.CheckpointSaver(args.save_dir,
                                 max_checkpoints=args.max_checkpoints,
                                 metric_name=args.metric_name,
                                 maximize_metric=args.maximize_metric,
                                 log=log)

    optimizer = optim.Adam(model.parameters(), args.lr,
                               weight_decay=args.l2_wd)

    log.info('Building dataset...')
    train_split = all_data['tiny']
    dev_split = all_data['tiny']

    if args.task == 'tag':
        train_dataset = SummarizationDataset(
                train_split['X'], train_split['y_tag'], train_split['gold_sums'])
        dev_dataset = SummarizationDataset(
                dev_split['X'], dev_split['y_tag'], dev_split['gold_sums'])
    else:
        train_dataset = SummarizationDataset(
                train_split['X'], train_split['y_decode'], train_split['gold_sums'])
        dev_dataset = SummarizationDataset(
                dev_split['X'], dev_split['y_decode'], dev_split['gold_sums'])

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

                y = y.float().to(device)
                if args.task == 'tag':
                    logits, mask = model(X)
                    loss = (F.binary_cross_entropy_with_logits(logits, y, reduction='none') * mask).mean()
                    loss_val = loss.item()
                else:
                    logits = model(X, y[:, :-1, :])
                    y_mask = (y[:, 1:].sum(-1, keepdim=True) != 0).float()
                    loss = (F.binary_cross_entropy_with_logits(logits, y[:, 1:], reduction='none') * y_mask).mean()
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
                    results, pred_dict = evaluate(args, model, dev_loader, device)
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

def evaluate(args, model, data_loader, device):
    model.eval()
    test_model = None
    if args.task == 'decode':
        test_model = GreedyDecoder(model, device)
        test_model.to(device)
        test_model.eval()
    all_preds = []
    gold_summaries = [] # tokenized gold summaries
    with torch.no_grad(), \
            tqdm(total=len(data_loader.dataset)) as progress_bar:
        for X, y, gold_sums in data_loader:
            X = X.to(device)
            # Setup for forward
            batch_size = X.size(0)
            y = y.float().to(device)
            if args.task == 'tag':
                logits, mask = model(X)
                loss = (F.binary_cross_entropy_with_logits(logits, y, reduction='none') * mask).mean()
            else:
                ## need to implement beam search
                token_ids, logits = test_model(X, max_tgt_len=y.size(1)-1)
                y_mask = (y[:, 1:].sum(-1, keepdim=True) != 0).float()
                loss = (F.binary_cross_entropy_with_logits(logits, y[:, 1:], reduction='none') * y_mask).mean()

            # Log info
            progress_bar.update(batch_size)
            progress_bar.set_postfix(Loss=loss.item())

            if args.task == 'tag':
                preds = util.untokenize(X, logits, mask, topk=60)
            else:
                assert test_model != None
                preds = util.unidize(token_ids)

            all_preds.extend(preds)
            gold_summaries.extend(gold_sums)

    model.train()

    valid_ids = [i for i in range(len(all_preds)) \
            if len(all_preds[i]) > 0 and all_preds[i][0] != '.']
    if len(valid_ids) == 0:
        return None, None
    pred = [all_preds[i] for i in valid_ids]
    ref = [gold_summaries[i] for i in valid_ids]
    rouge = Rouge()
    results = rouge.get_scores(pred, ref, avg=True)

    return results, all_preds

def test(args):
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-seed", default=827)
    parser.add_argument("-load_path", default=None)
    parser.add_argument("-split", default='tiny')
    parser.add_argument("-batch_size", default=8, type=int)
    parser.add_argument("-gpu_ids", default='0')
    parser.add_argument("-num_workers", default=1)
    parser.add_argument("-lr", default=0.0001)
    parser.add_argument("-l2_wd", default=0)
    parser.add_argument("-eval_steps", default=5000, type=int)
    parser.add_argument("-num_epochs", default=2)
    parser.add_argument("-max_grad_norm", default=2)
    parser.add_argument("-save_dir", default='saved_models')
    parser.add_argument("-name", default='default')
    parser.add_argument("-metric_name", default='rouge-1')
    parser.add_argument("-task", default='tag', choices=['tag', 'decode'])
    parser.add_argument("-max_checkpoints", default=3)
    parser.add_argument("-maximize_metric", default=True)
    args = parser.parse_args()

    if args.split == 'train' or args.split == 'tiny':
        train(args)
    else:
        test(args)
