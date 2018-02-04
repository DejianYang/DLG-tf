# - * - coding:utf-8 -*-

class BaseConfig(object):
    model = "BaseModel"

    #vocab size
    vocab_size = None
    sos_idx = None
    eos_idx = None

    # training options
    max_epoch = 20  # max training epochs
    batch_size = 16  # training batch size
    max_len = 100 # sentence max length
    infer_batch_size = 16  # infer batch size
    display_frequency = 100  # display_frequency
    checkpoint_frequency = 5000  # checkpoint frequency

    # optimizer configs
    optimizer = "adam"  # adam or sgd
    max_gradient_norm = 5.0  # gradient abs max cut
    learning_rate = 0.001  # initial learning rate
    start_decay_step = 10000
    decay_steps = 10000  # How frequent we decay
    decay_factor = 0.98  # How much we decay.

    # checkpoint max to keep
    max_to_keep = 20


class HREDConfig(BaseConfig):
    model = "HRED"

    prefix = "dialog"

    # model configs
    unit_type = "gru"  # gru or lstm
    emb_size = 500  # word embedding size
    enc_hidden_size = 1000  # encoder hidden size
    dec_hidden_size = 1000  # decoder hidden size
    num_layers = 1  # number of RNN layers
    dropout_keep_prob = 1.0  # Dropout keep_prob rate (not drop_prob)
    init_w = 0.1  # init weight scale

    max_turn = 10

    #infer options
    beam_size = 5
    infer_max_len = 100
    length_penalty_weight = 0.0


class HREDTestConfig(HREDConfig):
    model = "HRED"

    prefix = "dialog"
    # model configs
    unit_type = "gru"  # gru or lstm
    emb_size = 100  # word embedding size
    enc_hidden_size = 300  # encoder hidden size
    dec_hidden_size = 300  # decoder hidden size
    num_layers = 1  # number of RNN layers
    dropout_keep_prob = 1.0  # Dropout keep_prob rate (not drop_prob)
    init_w = 0.1  # init weight scale

    max_turn = 10

    batch_size =  16
    display_frequency = 10  # display_frequency
    checkpoint_frequency = 10  # checkpoint frequency


