# -*- coding:utf-8 -*-
from utils.vocab import *
from utils.iterator import *

vocab = load_vocabulary('../data/ubuntu/vocab.txt')
print('vocab size', vocab.size)

data_iter = get_data_iter(vocab=vocab,
                          dialog_file='../data/ubuntu/test.dialog.txt',
                          batch_size=16,
                          max_turn=10,
                          max_len=100,
                          infer=False,
                          shuffle=False)

print("training samples:", data_iter.num_samples)
print("training batches:", data_iter.num_batches)

for batch_data in data_iter.next_batch():
    print(batch_data.dialog.shape)
    print(batch_data.dialog_length)
    print(batch_data.dialog)
    break
    pass