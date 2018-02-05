# -*- coding:utf-8 -*-
from utils.vocab import *
from utils.iterator import *

vocab = load_vocabulary('../data/ubuntu/vocab.dialog.txt')
print('vocab size', vocab.size)

data_iter = get_dialog_data_iter(vocab=vocab,
                          dialog_file='../data/ubuntu/valid.dialog.txt',
                          batch_size=16,
                          max_turn=10,
                          max_len=100,
                                 infer=False,
                                 shuffle=False,
                                 bucket_config=[(4, 16), (6, 16), (8, 16), (10, 16)])


num_samples = 0
for batch_data in data_iter.next_batch():
    assert batch_data.response_input.shape == batch_data.response_output.shape
    # print(batch_data.target_output.shape)
    num_samples += batch_data.context.shape[0]
    pass
print("num samples:", num_samples)

for batch_data in data_iter.next_batch():
    print(batch_data.context.shape)
    print(batch_data.context)
    print(batch_data.response_length)
    print(batch_data.response_input)
    print(batch_data.response_output)
    break