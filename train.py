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

from models import SummarizerLinear, SummarizerAbstractive, GreedyDecoder, SummarizerLinearAttended, SummarizerRNN
import util

PROCESSED_DATA = os.path.join('data', 'data.pk')
PROCESSED_DATA_SUPER_TINY = os.path.join('data', 'super_tiny.pk')
PAD_VALUE = 0

class SummarizationDataset(data.Dataset):
    def __init__(self, X, y, gold_sums):
        super(SummarizationDataset, self).__init__()
        assert len(X) == len(y) == len(gold_sums)
        self.X = X
        self.y = y
        self.gold_sums = gold_sums
    def __getitem__(self, i):
        """
        We pad the y values to a total of 110 tokens. We pad with -1
        """
        y = [self.y[i][j] if j < len(self.y[i]) else -1 for j in range(110)]
        return (self.X[i], y, self.gold_sums[i])
    def __len__(self):
        return len(self.X)

def tag_collate_fn(examples):
    """
    This collate function corresponds to the tagging task
    """
    X, y, gold_sums = zip(*examples)

    ## First merge the X's
    lengths = [len(x) for x in X]
    max_len = max(lengths)
    padded_X = torch.zeros(len(X), max_len, dtype=torch.int64)
    for i, seq in enumerate(X):
        end = lengths[i]
        padded_X[i, :end] = torch.tensor(np.array(seq, dtype=np.int64))[:end]

    ## Then create the tagging mask
    index_tensor = torch.tensor(y, dtype=torch.int64)
    index_tensor[index_tensor == -1] = 0
    target = torch.zeros(len(X), max_len, dtype=torch.float32).scatter_(1, index_tensor, 1)
    return padded_X, target, gold_sums

def decode_collate_fn(examples):
    """
    This collate function corresponds to the decoding task
    """
    X, y, gold_sums = zip(*examples)

    ## First merge the X's
    lengths = [len(x) for x in X]
    max_len = max(lengths)
    padded_X = torch.zeros(len(X), max_len, dtype=torch.int64)
    for i, seq in enumerate(X):
        end = lengths[i]
        padded_X[i, :end] = torch.tensor(np.array(seq, dtype=np.int64))[:end]

    ## Then create the tagging mask
    target = torch.tensor(y, dtype=torch.float32)
    return padded_X, target, gold_sums

def train(args):
    args.save_dir = util.get_save_dir(args.save_dir, args.name, training=True)
    log = util.get_logger(args.save_dir, args.name)
    tbx = SummaryWriter(args.save_dir)
    if args.gpu_ids == 'cpu':
        device, args.gpu_ids = torch.device('cpu'), []
    else:
        device, args.gpu_ids = util.get_available_devices()
    log.info('training on device {} with gpu_id {}'.format(str(device), str(args.gpu_ids)))

    # Set random seed
    log.info('Using random seed {}...'.format(args.seed))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    log.info('Building model...')
    if args.task == 'tag':
        model = SummarizerLinear()
#        model = SummarizerLinearAttended(128, 256)
#        model = SummarizerRNN(128, 256)
    else:
        model = SummarizerAbstractive(128, 256, device)
    if len(args.gpu_ids) > 0:
        model = nn.DataParallel(model, args.gpu_ids)
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
    data_path = PROCESSED_DATA_SUPER_TINY if args.split == 'super_tiny' else PROCESSED_DATA
    with open(data_path, 'rb') as f:
        all_data = pickle.load(f)
    if 'tiny' in args.split:
        train_split = all_data['tiny']
        dev_split = all_data['tiny']
    else:
        train_split = all_data['train']
        dev_split = all_data['dev']
    train_dataset = SummarizationDataset(
            train_split['X'], train_split['y'], train_split['gold'])
    dev_dataset = SummarizationDataset(
            dev_split['X'], dev_split['y'], dev_split['gold'])
    collate_fn = tag_collate_fn if args.task == 'tag' else decode_collate_fn
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
        batch_num = 0
        with torch.enable_grad(), \
                tqdm(total=len(train_loader.dataset)) as progress_bar:
            for X, y, _ in train_loader:
                batch_size = X.size(0)
                batch_num += 1
                X = X.to(device)
                y = y.float().to(device) # (batch_size, max_len) for tag, (batch_size, 110) for decode
                optimizer.zero_grad()
                if args.task == 'tag':
                    logits = model(X) # (batch_size, max_len)
                    mask = (X != PAD_VALUE).float() # 1 for real data, 0 for pad, size of (batch_size, max_len)
                    loss = (F.binary_cross_entropy_with_logits(logits, y, reduction='none') * mask).mean()
                    loss_val = loss.item()
                else:
                    logits = model(X, y[:, :-1]) # (batch_size, 109, max_len)
                    loss = sum(F.cross_entropy(logits[i], y[i, 1:], ignore_index=-1, reduction='mean')\
                               for i in range(batch_size)) / batch_size
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
                    saver.save(step, model, results[args.metric_name], device)

#                     # Log to console
                    results_str = ', '.join('{}: {:05.2f}'.format(k, v)
                                            for k, v in results.items())
                    log.info('Dev {}'.format(results_str))

                    # Log to TensorBoard
                    log.info('Visualizing in TensorBoard...')
                    for k, v in results.items():
                        tbx.add_scalar('dev/{}'.format(k), v, step)
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
        ## need to implement beam search
        test_model = GreedyDecoder(model, device)
        test_model.to(device)
        test_model.eval()
    all_preds = []
    gold_summaries = [] # gold summaries
    total_loss = 0.0
    with torch.no_grad(), \
            tqdm(total=len(data_loader.dataset)) as progress_bar:
        for X, y, gold_sums in data_loader:
            X = X.to(device)
            batch_size = X.size(0)
            y = y.float().to(device)

            # Setup for forward
            if args.task == 'tag':
                logits = model(X) # (batch_size, max_len)
                mask = (X != PAD_VALUE).float() # 1 for real data, 0 for pad, size of (batch_size, max_len)
                loss = (F.binary_cross_entropy_with_logits(logits, y, reduction='none') * mask).mean()
            else:
                logits = test_model(X, max_tgt_len=y.size(1)-1) # (batch_size, 109, max_len)
                loss = sum(F.cross_entropy(logits[i], y[i, 1:], ignore_index=-1, reduction='mean')\
                               for i in range(batch_size)) / batch_size

            # Log info
            progress_bar.update(batch_size)
            progress_bar.set_postfix(Loss=loss.item())
            total_loss += loss.item()

            if args.task == 'tag':
                preds = util.tag_to_sents(X, logits, topk=60)
            else:
                preds = util.decode_to_sents(X, logits)

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
    results = {key: val['r'] for key, val in results.items()}
    results['total_loss'] = total_loss
    return results, pred

def test(args):
    args.save_dir = util.get_save_dir(args.save_dir, args.name, training=False)
    log = util.get_logger(args.save_dir, args.name)
    tbx = SummaryWriter(args.save_dir)
    if args.gpu_ids == 'cpu':
        device, args.gpu_ids = torch.device('cpu'), []
    else:
        device, args.gpu_ids = util.get_available_devices()
    log.info('testing on device {} with gpu_id {}'.format(str(device), str(args.gpu_ids)))
    log.info('Building model...')
    if args.task == 'tag':
        model = SummarizerLinear()
#        model = SummarizerLinearAttended(128, 256)
#        model = SummarizerRNN(128, 256)
    else:
        model = SummarizerAbstractive(128, 256, device)

    if len(args.gpu_ids) > 0:
        model = nn.DataParallel(model, args.gpu_ids)
    if args.load_path:
        log.info('Loading checkpoint from {}...'.format(args.load_path))
        model, step = util.load_model(model, args.load_path, args.gpu_ids)
    else:
        raise Exception('no specified checkpoint, abort')
    model = model.to(device)
    model.eval()
    log.info('Building dataset...')
    data_path = PROCESSED_DATA_SUPER_TINY if 'super_tiny' in args.split else PROCESSED_DATA
    with open(data_path, 'rb') as f:
        all_data = pickle.load(f)
    if 'tiny' in args.split:
        test_split = all_data['tiny']
    else:
        test_split = all_data['test']
    test_dataset = SummarizationDataset(
            test_split['X'], test_split['y'], test_split['gold'])
    collate_fn = tag_collate_fn if args.task == 'tag' else decode_collate_fn
    test_loader = data.DataLoader(test_dataset,
                                   batch_size=args.batch_size,
                                   num_workers=args.num_workers,
                                   shuffle=False,
                                   collate_fn=collate_fn)
    # Evaluate
    log.info('Evaluating at step {}...'.format(step))
    results, pred = evaluate(args, model, test_loader, device)
    if results is None:
        log.info('Selected predicted no select for all in batch')
        raise Exception('no results found')
    print(results)
    with open(os.path.join(args.save_dir, 'preds.txt'), 'w') as f:
        f.writelines(pred)
    with open(os.path.join(args.save_dir, 'result.txt'), 'w') as f:
        f.write(str(results))

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
    parser.add_argument("-eval_steps", default=50000, type=int)
    parser.add_argument("-num_epochs", default=15, type=int)
    parser.add_argument("-max_grad_norm", default=2)
    parser.add_argument("-save_dir", default='saved_models')
    parser.add_argument("-name", default='default')
    parser.add_argument("-metric_name", default='total_loss')
    parser.add_argument("-task", default='tag', choices=['tag', 'decode'])
    parser.add_argument("-max_checkpoints", default=3)
    parser.add_argument("-maximize_metric", default=False)
    args = parser.parse_args()

    if 'test' in args.split:
        test(args)
    else:
        train(args)
