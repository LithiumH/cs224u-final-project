"""Utility classes and methods.

Author:
    Chris Chute (chute@stanford.edu)
    modified by Haojun Li (haojun@stanford.edu)
"""
import logging
import os
import queue
import re
import shutil
import string
import torch
import torch.nn.functional as F
import torch.utils.data as data
import tqdm
import numpy as np
import json as json

from collections import Counter

from pytorch_pretrained_bert import BertTokenizer

class CheckpointSaver:
    """Class to save and load model checkpoints.

    Save the best checkpoints as measured by a metric value passed into the
    `save` method. Overwrite checkpoints with better checkpoints once
    `max_checkpoints` have been saved.

    Args:
        save_dir (str): Directory to save checkpoints.
        max_checkpoints (int): Maximum number of checkpoints to keep before
            overwriting old ones.
        metric_name (str): Name of metric used to determine best model.
        maximize_metric (bool): If true, best checkpoint is that which maximizes
            the metric value passed in via `save`. Otherwise, best checkpoint
            minimizes the metric.
        log (logging.Logger): Optional logger for printing information.
    """
    def __init__(self, save_dir, max_checkpoints, metric_name,
                 maximize_metric=False, log=None):
        super(CheckpointSaver, self).__init__()

        self.save_dir = save_dir
        self.max_checkpoints = max_checkpoints
        self.metric_name = metric_name
        self.maximize_metric = maximize_metric
        self.best_val = None
        self.ckpt_paths = queue.PriorityQueue()
        self.log = log
        self._print('Saver will {}imize {}...'
                    .format('max' if maximize_metric else 'min', metric_name))

    def is_best(self, metric_val):
        """Check whether `metric_val` is the best seen so far.

        Args:
            metric_val (float): Metric value to compare to prior checkpoints.
        """
        if metric_val is None:
            # No metric reported
            return False

        if self.best_val is None:
            # No checkpoint saved yet
            return True

        return ((self.maximize_metric and self.best_val < metric_val)
                or (not self.maximize_metric and self.best_val > metric_val))

    def _print(self, message):
        """Print a message if logging is enabled."""
        if self.log is not None:
            self.log.info(message)

    def save(self, step, model, metric_val, device):
        """Save model parameters to disk.

        Args:
            step (int): Total number of examples seen during training so far.
            model (torch.nn.DataParallel): Model to save.
            metric_val (float): Determines whether checkpoint is best so far.
            device (torch.device): Device where model resides.
        """
        ckpt_dict = {
            'model_name': model.__class__.__name__,
            'model_state': model.cpu().state_dict(),
            'step': step
        }
        model.to(device)

        checkpoint_path = os.path.join(self.save_dir,
                                       'step_{}.pth.tar'.format(step))
        torch.save(ckpt_dict, checkpoint_path)
        self._print('Saved checkpoint: {}'.format(checkpoint_path))

        if self.is_best(metric_val):
            # Save the best model
            self.best_val = metric_val
            best_path = os.path.join(self.save_dir, 'best.pth.tar')
            shutil.copy(checkpoint_path, best_path)
            self._print('New best checkpoint at step {}...'.format(step))

        # Add checkpoint path to priority queue (lowest priority removed first)
        if self.maximize_metric:
            priority_order = metric_val
        else:
            priority_order = -metric_val

        self.ckpt_paths.put((priority_order, checkpoint_path))

        # Remove a checkpoint if more than max_checkpoints have been saved
        if self.ckpt_paths.qsize() > self.max_checkpoints:
            _, worst_ckpt = self.ckpt_paths.get()
            try:
                os.remove(worst_ckpt)
                self._print('Removed checkpoint: {}'.format(worst_ckpt))
            except OSError:
                # Avoid crashing if checkpoint has been removed or protected
                pass


def load_model(model, checkpoint_path, gpu_ids, return_step=True):
    """Load model parameters from disk.

    Args:
        model (torch.nn.DataParallel): Load parameters into this model.
        checkpoint_path (str): Path to checkpoint to load.
        gpu_ids (list): GPU IDs for DataParallel.
        return_step (bool): Also return the step at which checkpoint was saved.

    Returns:
        model (torch.nn.DataParallel): Model loaded from checkpoint.
        step (int): Step at which checkpoint was saved. Only if `return_step`.
    """
    device = 'cuda:{}'.format(gpu_ids[0]) if gpu_ids else 'cpu'
    ckpt_dict = torch.load(checkpoint_path, map_location=device)

    # Build model, load parameters
    model.load_state_dict(ckpt_dict['model_state'])

    if return_step:
        step = ckpt_dict['step']
        return model, step

    return model


def get_available_devices():
    """Get IDs of all available GPUs.

    Returns:
        device (torch.device): Main device (GPU 0 or CPU).
        gpu_ids (list): List of IDs of all GPUs that are available.
    """
    gpu_ids = []
    if torch.cuda.is_available():
        gpu_ids += [gpu_id for gpu_id in range(torch.cuda.device_count())]
        device = torch.device('cuda:{}'.format(gpu_ids[0]))
        torch.cuda.set_device(device)
    else:
        device = torch.device('cpu')

    return device, gpu_ids


def masked_softmax(logits, mask, dim=-1, log_softmax=False):
    """Take the softmax of `logits` over given dimension, and set
    entries to 0 wherever `mask` is 0.

    Args:
        logits (torch.Tensor): Inputs to the softmax function.
        mask (torch.Tensor): Same shape as `logits`, with 0 indicating
            positions that should be assigned 0 probability in the output.
        dim (int): Dimension over which to take softmax.
        log_softmax (bool): Take log-softmax rather than regular softmax.
            E.g., some PyTorch functions such as `F.nll_loss` expect log-softmax.

    Returns:
        probs (torch.Tensor): Result of taking masked softmax over the logits.
    """
    mask = mask.type(torch.float32)
    masked_logits = mask * logits + (1 - mask) * -1e30
    softmax_fn = F.log_softmax if log_softmax else F.softmax
    probs = softmax_fn(masked_logits, dim)

    return probs


def visualize(tbx, pred_dict, eval_path, step, split, num_visuals):
    """Visualize text examples to TensorBoard.

    Args:
        tbx (tensorboardX.SummaryWriter): Summary writer.
        pred_dict (dict): dict of predictions of the form id -> pred.
        eval_path (str): Path to eval JSON file.
        step (int): Number of examples seen so far during training.
        split (str): Name of data split being visualized.
        num_visuals (int): Number of visuals to select at random from preds.
    """
    if num_visuals <= 0:
        return
    if num_visuals > len(pred_dict):
        num_visuals = len(pred_dict)

    visual_ids = np.random.choice(list(pred_dict), size=num_visuals, replace=False)

    with open(eval_path, 'r') as eval_file:
        eval_dict = json.load(eval_file)
    for i, id_ in enumerate(visual_ids):
        pred = pred_dict[id_] or 'N/A'
        example = eval_dict[str(id_)]
        question = example['question']
        context = example['context']
        answers = example['answers']

        gold = answers[0] if answers else 'N/A'
        tbl_fmt = ('- **Question:** {}\n'
                   + '- **Context:** {}\n'
                   + '- **Answer:** {}\n'
                   + '- **Prediction:** {}')
        tbx.add_text(tag='{}/{}_of_{}'.format(split, i + 1, num_visuals),
                     text_string=tbl_fmt.format(question, context, gold, pred),
                     global_step=step)


def save_preds(preds, save_dir, file_name='predictions.csv'):
    """Save predictions `preds` to a CSV file named `file_name` in `save_dir`.

    Args:
        preds (list): List of predictions each of the form (id, start, end),
            where id is an example ID, and start/end are indices in the context.
        save_dir (str): Directory in which to save the predictions file.
        file_name (str): File name for the CSV file.

    Returns:
        save_path (str): Path where CSV file was saved.
    """
    # Validate format
    if (not isinstance(preds, list)
            or any(not isinstance(p, tuple) or len(p) != 3 for p in preds)):
        raise ValueError('preds must be a list of tuples (id, start, end)')

    # Make sure predictions are sorted by ID
    preds = sorted(preds, key=lambda p: p[0])

    # Save to a CSV file
    save_path = os.path.join(save_dir, file_name)
    np.savetxt(save_path, np.array(preds), delimiter=',', fmt='%d')

    return save_path


def get_save_dir(base_dir, name, training, id_max=100):
    """Get a unique save directory by appending the smallest positive integer
    `id < id_max` that is not already taken (i.e., no dir exists with that id).

    Args:
        base_dir (str): Base directory in which to make save directories.
        name (str): Name to identify this training run. Need not be unique.
        training (bool): Save dir. is for training (determines subdirectory).
        id_max (int): Maximum ID number before raising an exception.

    Returns:
        save_dir (str): Path to a new directory with a unique name.
    """
    for uid in range(1, id_max):
        subdir = 'train' if training else 'test'
        save_dir = os.path.join(base_dir, subdir, '{}-{:02d}'.format(name, uid))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            return save_dir

    raise RuntimeError('Too many save directories created with the same name. \
                       Delete old save directories or use another name.')


def get_logger(log_dir, name):
    """Get a `logging.Logger` instance that prints to the console
    and an auxiliary file.

    Args:
        log_dir (str): Directory in which to create the log file.
        name (str): Name to identify the logs.

    Returns:
        logger (logging.Logger): Logger instance for logging events.
    """
    class StreamHandlerWithTQDM(logging.Handler):
        """Let `logging` print without breaking `tqdm` progress bars.

        See Also:
            > https://stackoverflow.com/questions/38543506
        """
        def emit(self, record):
            try:
                msg = self.format(record)
                tqdm.tqdm.write(msg)
                self.flush()
            except (KeyboardInterrupt, SystemExit):
                raise
            except:
                self.handleError(record)

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Log everything (i.e., DEBUG level and above) to a file
    log_path = os.path.join(log_dir, 'log.txt')
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.DEBUG)

    # Log everything except DEBUG level (i.e., INFO level and above) to console
    console_handler = StreamHandlerWithTQDM()
    console_handler.setLevel(logging.INFO)

    # Create format for the logs
    file_formatter = logging.Formatter('[%(asctime)s] %(message)s',
                                       datefmt='%m.%d.%y %H:%M:%S')
    file_handler.setFormatter(file_formatter)
    console_formatter = logging.Formatter('[%(asctime)s] %(message)s',
                                          datefmt='%m.%d.%y %H:%M:%S')
    console_handler.setFormatter(console_formatter)

    # add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

def remove_bert_tokens(sent):
    return re.sub(r'( ##)|(\[CLS\] )|(\s*\[SEP\])','', sent)

PAD_VALUE = 0
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
def tag_to_sents(X, logits, threshold=0., topk=60, max_len=110):
    mask = (X == PAD_VALUE)
    logits[mask] = float('-inf')
    if topk:
        _, inds = logits.topk(topk, dim=-1)
        token_ids = torch.gather(X, -1, inds).cpu().numpy()
    else:
        predicted_mask = (logits > threshold)
        token_ids = [torch.masked_select(X[i], predicted_mask[i].byte()).cpu().numpy().tolist() \
                for i in range(logits.size(0))]
    sents = [' '.join(tokenizer.convert_ids_to_tokens(selected_ids[:max_len]))\
         for selected_ids in token_ids]
    sents = [remove_bert_tokens(sent) for sent in sents]
    return sents # (batch_size, 'a (string) summary')

def decode_to_sents(X, logits):
    _, inds = logits.topk(1, dim=-1)
    inds = inds.squeeze(-1) # (batch_size, 109) each corresponds to the location in X
    token_ids = torch.gather(X, -1, inds)
    sents = [remove_bert_tokens(' '.join(tokenizer.convert_ids_to_tokens(selected_ids))) \
             for selected_ids in token_ids]
    return sents

# def greedy_decode(decoder, decoder_hidden, encoder_outputs, target_tensor):
#     '''
#     https://github.com/budzianowski/PyTorch-Beam-Search-Decoding
#     :param target_tensor: target indexes tensor of shape [B, T] where B is the batch size and T is the maximum length of the output sentence
#     :param decoder_hidden: input tensor of shape [1, B, H] for start of the decoding
#     :param encoder_outputs: if you are using attention mechanism you can pass encoder outputs, [T, B, H] where T is the maximum length of input sentence
#     :return: decoded_batch
#     '''
#
#     batch_size, seq_len = target_tensor.size()
#     decoded_batch = torch.zeros((batch_size, MAX_LENGTH))
#     decoder_input = torch.LongTensor([[SOS_token] for _ in range(batch_size)], device=device)
#
#     for t in range(MAX_LENGTH):
#         decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_outputs)
#
#         topv, topi = decoder_output.data.topk(1)  # get candidates
#         topi = topi.view(-1)
#         decoded_batch[:, t] = topi
#
#         decoder_input = topi.detach().view(-1, 1)
#
#     return decoded_batch


# def torch_from_json(path, dtype=torch.float32):
#     """Load a PyTorch Tensor from a JSON file.

#     Args:
#         path (str): Path to the JSON file to load.
#         dtype (torch.dtype): Data type of loaded array.

#     Returns:
#         tensor (torch.Tensor): Tensor loaded from JSON file.
#     """
#     with open(path, 'r') as fh:
#         array = np.array(json.load(fh))

#     tensor = torch.from_numpy(array).type(dtype)

#     return tensor


# def discretize(p_start, p_end, max_len=15, no_answer=False):
#     """Discretize soft predictions to get start and end indices.

#     Choose the pair `(i, j)` of indices that maximizes `p1[i] * p2[j]`
#     subject to `i <= j` and `j - i + 1 <= max_len`.

#     Args:
#         p_start (torch.Tensor): Soft predictions for start index.
#             Shape (batch_size, context_len).
#         p_end (torch.Tensor): Soft predictions for end index.
#             Shape (batch_size, context_len).
#         max_len (int): Maximum length of the discretized prediction.
#             I.e., enforce that `preds[i, 1] - preds[i, 0] + 1 <= max_len`.
#         no_answer (bool): Treat 0-index as the no-answer prediction. Consider
#             a prediction no-answer if `preds[0, 0] * preds[0, 1]` is greater
#             than the probability assigned to the max-probability span.

#     Returns:
#         start_idxs (torch.Tensor): Hard predictions for start index.
#             Shape (batch_size,)
#         end_idxs (torch.Tensor): Hard predictions for end index.
#             Shape (batch_size,)
#     """
#     if p_start.min() < 0 or p_start.max() > 1 \
#             or p_end.min() < 0 or p_end.max() > 1:
#         raise ValueError('Expected p_start and p_end to have values in [0, 1]')

#     # Compute pairwise probabilities
#     p_start = p_start.unsqueeze(dim=2)
#     p_end = p_end.unsqueeze(dim=1)
#     p_joint = torch.matmul(p_start, p_end)  # (batch_size, c_len, c_len)

#     # Restrict to pairs (i, j) such that i <= j <= i + max_len - 1
#     c_len, device = p_start.size(1), p_start.device
#     is_legal_pair = torch.triu(torch.ones((c_len, c_len), device=device))
#     is_legal_pair -= torch.triu(torch.ones((c_len, c_len), device=device),
#                                 diagonal=max_len)
#     if no_answer:
#         # Index 0 is no-answer
#         p_no_answer = p_joint[:, 0, 0].clone()
#         is_legal_pair[0, :] = 0
#         is_legal_pair[:, 0] = 0
#     else:
#         p_no_answer = None
#     p_joint *= is_legal_pair

#     # Take pair (i, j) that maximizes p_joint
#     max_in_row, _ = torch.max(p_joint, dim=2)
#     max_in_col, _ = torch.max(p_joint, dim=1)
#     start_idxs = torch.argmax(max_in_row, dim=-1)
#     end_idxs = torch.argmax(max_in_col, dim=-1)

#     if no_answer:
#         # Predict no-answer whenever p_no_answer > max_prob
#         max_prob, _ = torch.max(max_in_col, dim=-1)
#         start_idxs[p_no_answer > max_prob] = 0
#         end_idxs[p_no_answer > max_prob] = 0

#     return start_idxs, end_idxs


# def convert_tokens(eval_dict, qa_id, y_start_list, y_end_list, no_answer):
#     """Convert predictions to tokens from the context.

#     Args:
#         eval_dict (dict): Dictionary with eval info for the dataset. This is
#             used to perform the mapping from IDs and indices to actual text.
#         qa_id (int): List of QA example IDs.
#         y_start_list (list): List of start predictions.
#         y_end_list (list): List of end predictions.
#         no_answer (bool): Questions can have no answer. E.g., SQuAD 2.0.

#     Returns:
#         pred_dict (dict): Dictionary index IDs -> predicted answer text.
#         sub_dict (dict): Dictionary UUIDs -> predicted answer text (submission).
#     """
#     pred_dict = {}
#     sub_dict = {}
#     for qid, y_start, y_end in zip(qa_id, y_start_list, y_end_list):
#         context = eval_dict[str(qid)]["context"]
#         spans = eval_dict[str(qid)]["spans"]
#         uuid = eval_dict[str(qid)]["uuid"]
#         if no_answer and (y_start == 0 or y_end == 0):
#             pred_dict[str(qid)] = ''
#             sub_dict[uuid] = ''
#         else:
#             if no_answer:
#                 y_start, y_end = y_start - 1, y_end - 1
#             start_idx = spans[y_start][0]
#             end_idx = spans[y_end][1]
#             pred_dict[str(qid)] = context[start_idx: end_idx]
#             sub_dict[uuid] = context[start_idx: end_idx]
#     return pred_dict, sub_dict


# def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
#     if not ground_truths:
#         return metric_fn(prediction, '')
#     scores_for_ground_truths = []
#     for ground_truth in ground_truths:
#         score = metric_fn(prediction, ground_truth)
#         scores_for_ground_truths.append(score)
#     return max(scores_for_ground_truths)


# def eval_dicts(gold_dict, pred_dict, no_answer):
#     avna = f1 = em = total = 0
#     for key, value in pred_dict.items():
#         total += 1
#         ground_truths = gold_dict[key]['answers']
#         prediction = value
#         em += metric_max_over_ground_truths(compute_em, prediction, ground_truths)
#         f1 += metric_max_over_ground_truths(compute_f1, prediction, ground_truths)
#         if no_answer:
#             avna += compute_avna(prediction, ground_truths)

#     eval_dict = {'EM': 100. * em / total,
#                  'F1': 100. * f1 / total}

#     if no_answer:
#         eval_dict['AvNA'] = 100. * avna / total

#     return eval_dict


# def compute_avna(prediction, ground_truths):
#     """Compute answer vs. no-answer accuracy."""
#     return float(bool(prediction) == bool(ground_truths))


# # All methods below this line are from the official SQuAD 2.0 eval script
# # https://worksheets.codalab.org/rest/bundles/0x6b567e1cf2e041ec80d7098f031c5c9e/contents/blob/
# def normalize_answer(s):
#     """Convert to lowercase and remove punctuation, articles and extra whitespace."""

#     def remove_articles(text):
#         regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
#         return re.sub(regex, ' ', text)

#     def white_space_fix(text):
#         return ' '.join(text.split())

#     def remove_punc(text):
#         exclude = set(string.punctuation)
#         return ''.join(ch for ch in text if ch not in exclude)

#     def lower(text):
#         return text.lower()

#     return white_space_fix(remove_articles(remove_punc(lower(s))))


# def get_tokens(s):
#     if not s:
#         return []
#     return normalize_answer(s).split()


# def compute_em(a_gold, a_pred):
#     return int(normalize_answer(a_gold) == normalize_answer(a_pred))


# def compute_f1(a_gold, a_pred):
#     gold_toks = get_tokens(a_gold)
#     pred_toks = get_tokens(a_pred)
#     common = Counter(gold_toks) & Counter(pred_toks)
#     num_same = sum(common.values())
#     if len(gold_toks) == 0 or len(pred_toks) == 0:
#         # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
#         return int(gold_toks == pred_toks)
#     if num_same == 0:
#         return 0
#     precision = 1.0 * num_same / len(pred_toks)
#     recall = 1.0 * num_same / len(gold_toks)
#     f1 = (2 * precision * recall) / (precision + recall)
#     return f1
