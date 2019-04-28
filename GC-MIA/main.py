import tensorflow as tf
import numpy as np
from utils import get_data_info, read_data, load_word_embeddings
from model import GC_MIN




tf.app.flags.DEFINE_integer('embedding_dim', 300, 'dimension of word embedding')
tf.app.flags.DEFINE_integer('batch_size', 128, 'number of example per batch')
tf.app.flags.DEFINE_integer('n_epoch', 100, 'number of epoch')
tf.app.flags.DEFINE_integer('n_hidden', 200, 'number of hidden unit')
tf.app.flags.DEFINE_integer('n_class', 3, 'number of distinct class')
tf.app.flags.DEFINE_integer('pre_processed', 1, 'Whether the data is pre-processed')
tf.app.flags.DEFINE_float('learning_rate', 0.001, 'learning rate')
tf.app.flags.DEFINE_float('l2_reg', 0.00001, 'l2 regularization')
tf.app.flags.DEFINE_float('dropout', 0.5, 'dropout')

tf.app.flags.DEFINE_string('embedding_fname', './data/glove.6B.300d.txt', 'embedding file name')
tf.app.flags.DEFINE_string('train_fname', 'data/laptop/train.txt', 'training file name')
tf.app.flags.DEFINE_string('test_fname', 'data/laptop/test.txt', 'testing file name')
tf.app.flags.DEFINE_string('data_info', './data/data_lap_info.txt', 'the file saving data information')
tf.app.flags.DEFINE_string('train_data', './data/train_lap_data.txt', 'the file saving training data')
tf.app.flags.DEFINE_string('test_data', './data/test_lap_data.txt', 'the file saving testing data')


FLAGS = tf.app.flags.FLAGS



def main(_):
    print('Loading data info ...')
    word2id, max_aspect_len, max_context_len = get_data_info(FLAGS.train_fname, FLAGS.test_fname, FLAGS.data_info, FLAGS.pre_processed)


    print('Loading training data and testing data ...')
    train_data = read_data(FLAGS.train_fname, word2id, max_aspect_len, max_context_len, FLAGS.train_data, FLAGS.pre_processed)
    test_data = read_data(FLAGS.test_fname, word2id, max_aspect_len, max_context_len, FLAGS.test_data, FLAGS.pre_processed)

    print('Loading pre-trained word vectors ...')
    word2vec = load_word_embeddings(FLAGS.embedding_fname, FLAGS.embedding_dim, word2id)

    with tf.Session() as sess:
        model = IAN(FLAGS,word2id,max_aspect_len,max_context_len,word2vec, sess)
        model.build_model()
        model.run(train_data, test_data)


if __name__ == '__main__':
    tf.app.run()
