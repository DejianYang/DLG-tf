# -*- coding:utf-8 -*-
import numpy as np
import collections
from utils.vocab import Vocabulary
from utils.misc_utils import *

class DialogBatchInput(
    collections.namedtuple("DialogBatchInput",
                           ("dialog",
                            "dialog_length"))):
    pass


class Iterator(object):
    def __init__(self, dialog_data, batch_size, sos_idx, eos_idx, pad_idx=0,
                 max_len=None, max_turn=None, infer=False, shuffle=False):
        self._dialog_data = dialog_data
        self.batch_size = batch_size
        self._sos_idx = sos_idx
        self._eos_idx = eos_idx
        self._pad_idx = pad_idx
        self._max_len = max_len
        self._max_turn = max_turn
        self._infer = infer
        self._shuffle = shuffle

        self.num_samples = len(self._dialog_data)
        self.num_batches = math.ceil(self.num_samples / batch_size)

    def shuffle(self, idx):
        np.random.shuffle(idx)

    def __len__(self):
        return self.num_batches

    def _dynamic_padding(self, batch_data, flag=False):
        turns = [len(d) for d in batch_data]
        sent_lengths = [[len(sent) for sent in d] for d in batch_data]

        max_turn = max(turns)
        max_length = max(max(sent_lengths))

        if self._max_turn is not None and self._max_turn < max_turn:
            max_turn = self._max_turn
        if self._max_len is not None and self._max_len < max_length:
            max_length = self._max_len

        pad_dialog_data, pad_dialog_lengths = [], []
        for turn, turn_data in zip(turns, batch_data):
            if turn > max_turn:
                turn_data = turn_data[-max_turn:]
            pad_sents = []
            pad_sent_lengths = []

            pad_sent_max_length = max_length + 2 if flag else max_length
            for sent in turn_data:
                if len(sent) > max_length:
                    sent = sent[:max_length]
                if flag: # add sos and eos idx for target
                    sent = [self._sos_idx] + sent + [self._eos_idx]

                pad_sents.append(sent + [self._pad_idx] * (pad_sent_max_length - len(sent)))
                pad_sent_lengths.append(len(sent))

            while len(pad_sents) < max_turn:
                pad_sents += [[self._pad_idx] * pad_sent_max_length]
                pad_sent_lengths += [0]

            pad_dialog_data += [pad_sents]
            pad_dialog_lengths += [pad_sent_lengths]

        pad_dialog_data = np.asarray(pad_dialog_data, dtype='int32')
        pad_sent_lengths = np.asarray(pad_dialog_lengths, dtype='int32')

        return pad_dialog_data, pad_sent_lengths

    def _get_one_mini_batch(self, batch_idx):
        batch_data = []
        for idx in batch_idx:
            sample = self._dialog_data[idx]
            batch_data += [sample]
        pad_batch, pad_length = self._dynamic_padding(batch_data, True)
        return DialogBatchInput(dialog=pad_batch,
                                dialog_length=pad_length)



    def next_batch(self):
        samples_ids = list(range(self.num_samples))
        if not self._infer and self._shuffle:
            self.shuffle(samples_ids)
        start_idx = 0
        while start_idx < self.num_samples:
            end_idx = start_idx + self.batch_size
            if end_idx > self.num_samples:
                end_idx = self.num_samples

            batch_idx = samples_ids[start_idx:end_idx]
            yield self._get_one_mini_batch(batch_idx)
            start_idx += self.batch_size


class DialogIterator(object):
    def __init__(self, dialog_data,
                 batch_size,
                 sos_idx,
                 eos_idx,
                 pad_idx=0,
                 max_len=None,
                 max_turn=None,
                 infer=False):

        self._dialog_data = dialog_data
        self._batch_size = batch_size
        self._sos_idx = sos_idx
        self._eos_idx = eos_idx
        self._pad_idx = pad_idx
        self._max_len = max_len
        self._max_turn = max_turn
        self._infer = infer

        # generate context response pairs
        self._context_response_pairs = self.generate_context_response_pairs(self._dialog_data)

        self.num_dialogs = len(self._dialog_data)
        self.num_samples = len(self._context_response_pairs)
        self.num_batches = math.ceil(self.num_samples / batch_size)

    def generate_context_response_pairs(self, dialog_data):
        ctx_resp_pairs = []
        ignore_samples = 0
        for dialog in dialog_data:
            turns = len(dialog)
            if turns < 2:
                ignore_samples += 1
                continue
            for i in range(turns-1):
                ii = 0 if i <= self._max_len else i - self._max_len
                ctx = dialog[ii:i+1]
                resp = dialog[i+1]
                ctx_resp_pairs.append((ctx, resp))

        print("generate context-response paris over; dialog samples:%d; pairs num:%d; ignored samples:%d" %
              (len(self._dialog_data), len(ctx_resp_pairs), ignore_samples))
        return ctx_resp_pairs

    def make_bucket_batches(self):
        pass


def get_dialog_data_iter(vocab:Vocabulary,
                         dialog_file,
                         batch_size,
                         max_len=None,
                         max_turn=None,
                         model="HRED",
                         infer=False,
                         shuffle=False):
    dialog_data = load_dialog_data(dialog_file, '</d>')
    dialog_idx = [[vocab.convert2idx(sent) for sent in dialog] for dialog in dialog_data]

    if model == 'HRED':
        data_iter = DialogIterator(dialog_data=dialog_idx,
                                   batch_size=batch_size,
                             sos_idx=vocab.sos_idx,
                             eos_idx=vocab.eos_idx,
                             pad_idx=vocab.pad_idx,
                             max_len=max_len,
                             max_turn=max_turn,
                             infer=infer)
        return data_iter
    else:
        raise NotImplementedError("Not Implemented Data Iterator")

def get_data_iter(vocab:Vocabulary,
                  dialog_file,
                  batch_size,
                  max_len=None,
                  max_turn=None,
                  model="HRED",
                  infer=False,
                  shuffle=False):

    dialog_data = load_dialog_data(dialog_file, '</d>')
    dialog_idx = [[vocab.convert2idx(sent) for sent in dialog] for dialog in dialog_data]

    if model == 'HRED':
        data_iter = Iterator(dialog_data=dialog_idx,
                             batch_size=batch_size,
                             sos_idx=vocab.sos_idx,
                             eos_idx=vocab.eos_idx,
                             pad_idx=vocab.pad_idx,
                             max_len=max_len,
                             max_turn=max_turn,
                             infer=infer,
                             shuffle=shuffle)
        return data_iter
    else:
        raise NotImplementedError("Not Implemented Data Iterator")

def get_train_iter(data_dir, vocab, config):

    train_file = os.path.join(data_dir, 'train.%s.txt' % config.prefix)
    train_iter = get_data_iter(vocab=vocab,
                               dialog_file=train_file,
                               batch_size=config.batch_size,
                               max_len=config.max_len,
                               max_turn=config.max_turn,
                               model=config.model,
                               infer=False,
                               shuffle=False)

    valid_file = os.path.join(data_dir, 'valid.%s.txt' % config.prefix)
    valid_iter = get_data_iter(vocab=vocab,
                               dialog_file=valid_file,
                               batch_size=config.batch_size,
                               max_len=config.max_len,
                               max_turn=config.max_turn,
                               model=config.model,
                               infer=False,
                               shuffle=True)

    test_file = os.path.join(data_dir, 'test.%s.txt' % config.prefix)
    test_iter = get_data_iter(vocab=vocab,
                               dialog_file=test_file,
                               batch_size=config.batch_size,
                               max_len=config.max_len,
                               max_turn=config.max_turn,
                               model=config.model,
                               infer=False,
                               shuffle=False)

    return train_iter, valid_iter, test_iter

def get_infer_iter(context_file, vocab, config):
    infer_iter = get_data_iter(vocab=vocab,
                               dialog_file=context_file,
                               batch_size=config.infer_batch_size,
                               max_len=config.max_len,
                               max_turn=config.max_turn,
                               model=config.model,
                               infer=True,
                               shuffle=False)
    return infer_iter

