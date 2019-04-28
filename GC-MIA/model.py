'''在原来袁婷模型的基础上加上注意力机制3的模型，注意力机制3：取消原来的平均池化，改成注意力机制，成为一个向量'''
import tensorflow as tf
import numpy as np
import time
from utils import get_batch_index
from funct import *

class GC_MIN(object):

    def __init__(self, config, word2id,max_aspect_len,max_context_len,word2vec,sess):
        self.embedding_dim = config.embedding_dim
        self.batch_size = config.batch_size
        self.n_epoch = config.n_epoch
        self.n_hidden = config.n_hidden
        self.n_class = config.n_class
        self.learning_rate = config.learning_rate
        self.l2_reg = config.l2_reg
        self.dropout = config.dropout

        self.word2id = word2id
        self.max_aspect_len = max_aspect_len
        self.max_context_len = max_context_len
        self.word2vec = word2vec
        self.sess = sess

    def build_model(self):
        with tf.name_scope('inputs'):
            self.aspects = tf.placeholder(tf.int32, [None, self.max_aspect_len])
            self.contexts = tf.placeholder(tf.int32, [None, self.max_context_len])
            self.labels = tf.placeholder(tf.int32, [None, self.n_class])
            self.aspect_lens = tf.placeholder(tf.int32, None)
            self.context_lens = tf.placeholder(tf.int32, None)
            self.dropout_keep_prob = tf.placeholder(tf.float32)

            aspect_inputs = tf.nn.embedding_lookup(self.word2vec, self.aspects)
            aspect_inputs = tf.cast(aspect_inputs, tf.float32)
            aspect_inputs = tf.nn.dropout(aspect_inputs, keep_prob=self.dropout_keep_prob)

            context_inputs = tf.nn.embedding_lookup(self.word2vec, self.contexts)
            context_inputs = tf.cast(context_inputs, tf.float32)
            context_inputs = tf.nn.dropout(context_inputs, keep_prob=self.dropout_keep_prob)

        with tf.name_scope('weights'):
            weights = {
                'aspect_score': tf.get_variable(
                    name='W_a',
                    shape=[self.max_aspect_len, self.n_hidden],
                    initializer=tf.random_uniform_initializer(-0.1, 0.1),
                    regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg)
                ),
                'context_score': tf.get_variable(
                    name='W_c',
                    shape=[self.max_context_len, self.n_hidden],
                    initializer=tf.random_uniform_initializer(-0.1, 0.1),
                    regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg)
                ),
                'aspect_score2': tf.get_variable(
                    name='W_a2',
                    shape=[self.n_hidden, self.n_hidden],
                    initializer=tf.random_uniform_initializer(-0.1, 0.1),
                    regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg)
                ),
                'context_score2': tf.get_variable(
                    name='W_c2',
                    shape=[self.n_hidden, self.n_hidden],
                    initializer=tf.random_uniform_initializer(-0.1, 0.1),
                    regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg)
                ),
                'aspect_score3': tf.get_variable(
                    name='W_a3',
                    shape=[self.n_hidden, self.n_hidden],
                    initializer=tf.random_uniform_initializer(-0.1, 0.1),
                    regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg)
                ),
                'context_score3': tf.get_variable(
                    name='W_c3',
                    shape=[self.n_hidden, self.n_hidden],
                    initializer=tf.random_uniform_initializer(-0.1, 0.1),
                    regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg)
                ),
                'softmax': tf.get_variable(
                    name='W_l',
                    shape=[self.n_hidden * 2, self.n_class],
                    initializer=tf.random_uniform_initializer(-0.1, 0.1),
                    regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg)
                ),
            }

        with tf.name_scope('biases'):
            biases = {
                'aspect_score': tf.get_variable(
                    name='B_a',
                    shape=[self.max_aspect_len, 1],
                    initializer=tf.zeros_initializer(),
                    regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg)
                ),
                'context_score': tf.get_variable(
                    name='B_c',
                    shape=[self.max_context_len, 1],
                    initializer=tf.zeros_initializer(),
                    regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg)
                ),
                'aspect_score2': tf.get_variable(
                    name='B_a2',
                    shape=[self.max_aspect_len, 1],
                    initializer=tf.zeros_initializer(),
                    regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg)
                ),
                'context_score2': tf.get_variable(
                    name='B_c2',
                    shape=[self.max_context_len, 1],
                    initializer=tf.zeros_initializer(),
                    regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg)
                ),
                'aspect_score3': tf.get_variable(
                    name='B_a3',
                    shape=[self.max_aspect_len, 1],
                    initializer=tf.zeros_initializer(),
                    regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg)
                ),
                'context_score3': tf.get_variable(
                    name='B_c3',
                    shape=[self.max_context_len, 1],
                    initializer=tf.zeros_initializer(),
                    regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg)
                ),

                'softmax': tf.get_variable(
                    name='B_l',
                    shape=[self.n_class],
                    initializer=tf.zeros_initializer(),
                    regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg)
                ),
            }

        with tf.name_scope('dynamic_rnn'):
            aspect_outputs, aspect_state = tf.nn.dynamic_rnn(
                tf.contrib.rnn.LSTMCell(self.n_hidden),
                inputs=aspect_inputs,
                sequence_length=self.aspect_lens,
                dtype=tf.float32,
                scope='aspect_lstm'
            )

            context_outputs, context_state = tf.nn.dynamic_rnn(
                tf.contrib.rnn.LSTMCell(self.n_hidden),
                inputs=context_inputs,
                sequence_length=self.context_lens,
                dtype=tf.float32,
                scope='context_lstm'
            )

            # 横着求平均
            att_mat = tf.nn.softmax(tf.nn.tanh(
                tf.einsum('aij,ajk->aik', tf.einsum("ijk,kl->ijl", aspect_outputs, weights['aspect_score2']), tf.matrix_transpose(context_outputs))))
            print('att_mat', att_mat.get_shape())
            self.aspect_reps = tf.einsum('aij,ajk->aik', tf.matrix_transpose(att_mat), aspect_outputs)
            print('self.aspect_reps', self.aspect_reps.get_shape())
            self.context_reps = tf.einsum('aij,ajk->aik', att_mat, context_outputs)
            print(' self.context_reps', self.context_reps.get_shape())
            #横着求平均


            with tf.name_scope('dynamic_rnn2'):
                aspect_outputs2, aspect_state2 = tf.nn.dynamic_rnn(
                    tf.contrib.rnn.LSTMCell(self.n_hidden),
                    inputs=self.context_reps,
                    sequence_length=self.aspect_lens,
                    dtype=tf.float32,
                    scope='aspect_lstm2'
                )
                print('aspect_outputs2', aspect_outputs2.get_shape())

                context_outputs2, context_state2 = tf.nn.dynamic_rnn(
                    tf.contrib.rnn.LSTMCell(self.n_hidden),
                    inputs=self.aspect_reps,
                    sequence_length=self.context_lens,
                    dtype=tf.float32,
                    scope='context_lstm2'
                )

                print('context_outputs2', context_outputs2.get_shape())

            x2_expanded = tf.expand_dims(tf.matrix_transpose(context_outputs2), -1)
            x1_expanded = tf.expand_dims(tf.matrix_transpose(aspect_outputs2), -1)

            print('x1_expanded', x1_expanded.get_shape())
            print(' x2_expanded', x2_expanded.get_shape)

            att_mat = make_attention_mat(x1_expanded, x2_expanded)

            print("att_mat", att_mat.get_shape())

            # [batch, s, s] * [s,d] => [batch, s, d]
            # matrix transpose => [batch, d, s]
            # expand dims => [batch, d, s, 1]
            # einsum乘法
            batch_size = tf.shape(context_outputs2)[0]
            context_avg_iter = tf.TensorArray(tf.float32, 1, dynamic_size=True, infer_shape=False)
            context_avg_iter = context_avg_iter.unstack(att_mat)
            a_asp = tf.TensorArray(size=batch_size, dtype=tf.float32)

            def body(i, a_asp):
                b = context_avg_iter.read(i)
                c = context_avg_iter.read(i)
                for j in range(self.max_context_len):
                    z = 0
                    for k in range(self.max_aspect_len):
                        z = z + b[k, j]

                    for k in range(self.max_aspect_len):
                        c[k, j] / z

                a_asp = a_asp.write(i, c)
                return (i + 1, a_asp)

            def condition(i, a_asp):
                return i < batch_size

            _, a_final = tf.while_loop(cond=condition, body=body, loop_vars=(0, a_asp))
            att_mat1 = tf.reshape(a_final.stack(), [-1, self.max_aspect_len, self.max_context_len])
            # [batch, s, s] * [s,d] => [batch, s, d]
            # matrix transpose => [batch, d, s]
            # expand dims => [batch, d, s, 1]
            # einsum乘法
            print('att_mat1', att_mat1.get_shape)
            x1_a = tf.einsum('aij,ajk->aik', tf.matrix_transpose(att_mat1), aspect_outputs2)
            print("x1_a", x1_a.get_shape())

            context_avg_iter = tf.TensorArray(tf.float32, 1, dynamic_size=True, infer_shape=False)
            context_avg_iter = context_avg_iter.unstack(att_mat)
            a_con = tf.TensorArray(size=batch_size, dtype=tf.float32)

            def body(i, a_con):
                b = context_avg_iter.read(i)
                c = context_avg_iter.read(i)
                for j in range(self.max_aspect_len):
                    z = 0
                    for k in range(self.max_context_len):
                        z = z + b[j, k]

                    for k in range(self.max_context_len):
                        c[j, k] / z

                a_con = a_con.write(i, c)
                return (i + 1, a_con)

            def condition(i, a_con):
                return i < batch_size

            _, a_final1 = tf.while_loop(cond=condition, body=body, loop_vars=(0, a_con))
            att_mat2 = tf.reshape(a_final1.stack(), [-1,self.max_aspect_len, self.max_context_len])
            x2_a = tf.einsum('aij,ajk->aik', att_mat2, context_outputs2)
            print("x2_a", x2_a.get_shape())


            aspect_outputs3, aspect_state3 = tf.nn.dynamic_rnn(
                tf.contrib.rnn.LSTMCell(self.n_hidden),
                inputs=x2_a,
                sequence_length=self.aspect_lens,
                dtype=tf.float32,
                scope='aspect_lstm3'
            )

            context_outputs3, context_state3 = tf.nn.dynamic_rnn(
                tf.contrib.rnn.LSTMCell(self.n_hidden),
                inputs=x1_a,
                sequence_length=self.context_lens,
                dtype=tf.float32,
                scope='context_lstm3'
            )


            batch_size = tf.shape(aspect_outputs3)[0]
            # 横着求平均
            #开始目标注意力机制了
            aspect_outputs_iter1 = tf.TensorArray(tf.float32, 1, dynamic_size=True, infer_shape=False)
            aspect_outputs_iter1 = aspect_outputs_iter1.unstack(aspect_outputs3)
            # context_avg_iter = tf.TensorArray(tf.float32, 1, dynamic_size=True, infer_shape=False)
            # context_avg_iter = context_avg_iter.unstack(context_avg)
            aspect_rep1 = tf.TensorArray(size=batch_size, dtype=tf.float32)
            aspect_att1 = tf.TensorArray(size=batch_size, dtype=tf.float32)
            print(aspect_outputs3.get_shape())

            def body(i, aspect_rep1, aspect_att1):
                a1 = aspect_outputs_iter1.read(i)
                # b = context_avg_iter.read(i)
                aspect_score1 = tf.nn.tanh(a1)
                print("aspect_score1",aspect_score1.get_shape())
                print('aspect_score1',aspect_score1.get_shape())
                aspect_att_temp1 = tf.nn.softmax(aspect_score1)
                #进行归一化
                # 这里的aspcet_sorce1的维度是24，300
                aspect_att1 = aspect_att1.write(i, aspect_att_temp1)
                aspect_rep1 = aspect_rep1.write(i, tf.multiply(aspect_att_temp1, a1))
                return (i + 1, aspect_rep1, aspect_att1)

            def condition(i, aspect_rep, aspect_att):
                return i < batch_size

            _, aspect_rep_final1, aspect_att_final1 = tf.while_loop(cond=condition, body=body,
                                                                  loop_vars=(0, aspect_rep1, aspect_att1))
            self.aspect_atts1 = tf.reshape(aspect_att_final1.stack(), [-1, self.max_aspect_len])
            self.aspect_reps1 = tf.reshape(aspect_rep_final1.stack(), [-1, self.n_hidden])
            #目标注意力机制结束
            aspect_avg = self.aspect_reps1
            # aspect_avg = tf.reduce_mean(aspect_outputs3, 1)
            print("aspect_outputs",aspect_outputs3.get_shape())
            #横向取平均
            #开始内容注意力机制了
            context_outputs_iter1 = tf.TensorArray(tf.float32, 1, dynamic_size=True, infer_shape=False)
            context_outputs_iter1 = context_outputs_iter1.unstack(context_outputs3)
            # aspect_avg_iter = tf.TensorArray(tf.float32, 1, dynamic_size=True, infer_shape=False)
            # aspect_avg_iter = aspect_avg_iter.unstack(aspect_avg)
            context_rep1 = tf.TensorArray(size=batch_size, dtype=tf.float32)
            context_att1 = tf.TensorArray(size=batch_size, dtype=tf.float32)

            def body(i, context_rep1, context_att1):
                a = context_outputs_iter1.read(i)
                # b = aspect_avg_iter.read(i)
                context_score1 = tf.nn.tanh(a)
                context_att_temp1 = tf.nn.softmax(context_score1)
                context_att1 = context_att1.write(i, context_att_temp1)
                context_rep1 = context_rep1.write(i, tf.multiply(context_att_temp1, a))
                return (i + 1, context_rep1, context_att1)

            def condition(i, context_rep1, context_att1):
                return i < batch_size

            _, context_rep_final1, context_att_final1 = tf.while_loop(cond=condition, body=body,
                                                                    loop_vars=(0, context_rep1, context_att1))
            self.context_atts1 = tf.reshape(context_att_final1.stack(), [-1, self.max_context_len])
            self.context_reps1 = tf.reshape(context_rep_final1.stack(), [-1, self.n_hidden])
            #内容注意力机制结束
            context_avg =self.context_reps1
            # context_avg = tf.reduce_mean(context_outputs3, 1)
            print("context_outputs3",context_outputs3.get_shape())
            aspect_outputs_iter = tf.TensorArray(tf.float32, 1, dynamic_size=True, infer_shape=False)
            aspect_outputs_iter = aspect_outputs_iter.unstack(aspect_outputs3)
            context_avg_iter = tf.TensorArray(tf.float32, 1, dynamic_size=True, infer_shape=False)
            context_avg_iter = context_avg_iter.unstack(context_avg)
            aspect_rep = tf.TensorArray(size=batch_size, dtype=tf.float32)
            aspect_att = tf.TensorArray(size=batch_size, dtype=tf.float32)

            def body(i, aspect_rep, aspect_att):
                a = aspect_outputs_iter.read(i)
                b = context_avg_iter.read(i)
                aspect_score = tf.reshape(tf.nn.tanh(tf.matmul(tf.matmul(a, weights['aspect_score3']), tf.reshape(b, [-1, 1])) + biases['aspect_score3']),[1, -1])
                aspect_att_temp = tf.nn.softmax(aspect_score)
                aspect_att = aspect_att.write(i, aspect_att_temp)
                aspect_rep = aspect_rep.write(i, tf.matmul(aspect_att_temp, a))
                return (i + 1, aspect_rep, aspect_att)

            def condition(i, aspect_rep, aspect_att):
                return i < batch_size

            _, aspect_rep_final, aspect_att_final = tf.while_loop(cond=condition, body=body,loop_vars=(0, aspect_rep, aspect_att))
            self.aspect_atts = tf.reshape(aspect_att_final.stack(), [-1, self.max_aspect_len])
            self.aspect_reps = tf.reshape(aspect_rep_final.stack(), [-1, self.n_hidden])

            context_outputs_iter = tf.TensorArray(tf.float32, 1, dynamic_size=True, infer_shape=False)
            context_outputs_iter = context_outputs_iter.unstack(context_outputs3)
            aspect_avg_iter = tf.TensorArray(tf.float32, 1, dynamic_size=True, infer_shape=False)
            aspect_avg_iter = aspect_avg_iter.unstack(aspect_avg)
            context_rep = tf.TensorArray(size=batch_size, dtype=tf.float32)
            context_att = tf.TensorArray(size=batch_size, dtype=tf.float32)

            def body(i, context_rep, context_att):
                a = context_outputs_iter.read(i)
                b = aspect_avg_iter.read(i)
                context_score = tf.reshape(tf.nn.tanh(
                    tf.matmul(tf.matmul(a, weights['context_score3']), tf.reshape(b, [-1, 1])) + biases[
                        'context_score3']), [1, -1])
                context_att_temp = tf.nn.softmax(context_score)
                context_att = context_att.write(i, context_att_temp)
                context_rep = context_rep.write(i, tf.matmul(context_att_temp, a))
                return (i + 1, context_rep, context_att)

            def condition(i, context_rep, context_att):
                return i < batch_size

            _, context_rep_final, context_att_final = tf.while_loop(cond=condition, body=body,
                                                                    loop_vars=(0, context_rep, context_att))
            self.context_atts = tf.reshape(context_att_final.stack(), [-1, self.max_context_len])
            self.context_reps = tf.reshape(context_rep_final.stack(), [-1, self.n_hidden])

            # 连起来
            self.reps = tf.concat([self.aspect_reps, self.context_reps], 1)

            self.predict =tf.matmul(self.reps, weights['softmax']) + biases['softmax']

        with tf.name_scope('loss'):
            self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = self.predict, labels = self.labels))
            self.global_step = tf.Variable(0, name="tr_global_step", trainable=False)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost, global_step=self.global_step)

        with tf.name_scope('predict'):
            self.correct_pred = tf.equal(tf.argmax(self.predict, 1), tf.argmax(self.labels, 1))
            self.accuracy = tf.reduce_sum(tf.cast(self.correct_pred, tf.int32))

        summary_loss = tf.summary.scalar('loss', self.cost)
        summary_acc = tf.summary.scalar('acc', self.accuracy)
        self.train_summary_op = tf.summary.merge([summary_loss, summary_acc])
        self.test_summary_op = tf.summary.merge([summary_loss, summary_acc])
        timestamp = str(int(time.time()))
        _dir = 'logs/' + str(timestamp) + '_r' + str(self.learning_rate) + '_b' + str(self.batch_size) + '_l' + str(self.l2_reg)
        self.train_summary_writer = tf.summary.FileWriter(_dir + '/train', self.sess.graph)
        self.test_summary_writer = tf.summary.FileWriter(_dir + '/test', self.sess.graph)

    def train(self, data):
        aspects, contexts, labels, aspect_lens, context_lens = data
        cost, cnt = 0., 0

        for sample, num in self.get_batch_data(aspects, contexts, labels, aspect_lens, context_lens, self.batch_size, True, self.dropout):
            _, loss, step, summary = self.sess.run([self.optimizer, self.cost, self.global_step, self.train_summary_op], feed_dict=sample)
            self.train_summary_writer.add_summary(summary, step)
            cost += loss * num
            cnt += num

        _, train_acc = self.test(data)
        return cost / cnt, train_acc

    def test(self, data):
        aspects, contexts, labels, aspect_lens, context_lens = data
        cost, acc, cnt = 0., 0, 0

        for sample, num in self.get_batch_data(aspects, contexts, labels, aspect_lens, context_lens, len(data), False, 1.0):
            loss, accuracy, step, summary = self.sess.run([self.cost, self.accuracy, self.global_step, self.test_summary_op], feed_dict=sample)
            cost += loss * num
            acc += accuracy
            cnt += num

        self.test_summary_writer.add_summary(summary, step)
        return cost / cnt, acc / cnt

    def analysis(self, train_data, test_data):
        timestamp = str(int(time.time()))

        aspects, contexts, labels, aspect_lens, context_lens = train_data
        with open('analysis/train_' + str(timestamp) + '.txt', 'w') as f:
            for sample, num in self.get_batch_data(aspects, contexts, labels, aspect_lens, context_lens, len(train_data), False, 1.0):
                aspect_atts, context_atts, correct_pred = self.sess.run([self.aspect_atts, self.context_atts, self.correct_pred], feed_dict=sample)
                for a, b, c in zip(aspect_atts, context_atts, correct_pred):
                    a = str(a).replace('\n', '')
                    b = str(b).replace('\n', '')
                    f.write('%s\n%s\n%s\n' % (a, b, c))
        print('Finishing analyzing training data')

        aspects, contexts, labels, aspect_lens, context_lens = test_data
        with open('analysis/test_' + str(timestamp) + '.txt', 'w') as f:
            for sample, num in self.get_batch_data(aspects, contexts, labels, aspect_lens, context_lens, len(test_data), False, 1.0):
                aspect_atts, context_atts, correct_pred = self.sess.run([self.aspect_atts, self.context_atts, self.correct_pred], feed_dict=sample)
                for a, b, c in zip(aspect_atts, context_atts, correct_pred):
                    a = str(a).replace('\n', '')
                    b = str(b).replace('\n', '')
                    f.write('%s\n%s\n%s\n' % (a, b, c))
        print('Finishing analyzing testing data')

    def run(self, train_data, test_data):
        saver = tf.train.Saver(tf.trainable_variables())

        print('Training ...')
        self.sess.run(tf.global_variables_initializer())
        max_acc, step = 0., -1
        for i in range(self.n_epoch):
            train_loss, train_acc = self.train(train_data)
            test_loss, test_acc = self.test(test_data)
            if test_acc > max_acc:
                max_acc = test_acc
                step = i
                saver.save(self.sess, 'models/model_iter', global_step=step)
            print('epoch %s: train-loss=%.6f; train-acc=%.6f; train-loss=%.6f; test-acc=%.6f;' % (str(i), train_loss, train_acc, test_loss, test_acc))
        saver.save(self.sess, 'models/model_final')
        print('The max accuracy of testing results is %s of step %s' % (max_acc, step))

        print('Analyzing ...')
        saver.restore(self.sess, tf.train.latest_checkpoint('models/'))
        self.analysis(train_data, test_data)

    def get_batch_data(self, aspects, contexts, labels, aspect_lens, context_lens, batch_size, is_shuffle, keep_prob):
        for index in get_batch_index(len(aspects), batch_size, is_shuffle):
            feed_dict = {
                self.aspects: aspects[index],
                self.contexts: contexts[index],
                self.labels: labels[index],
                self.aspect_lens: aspect_lens[index],
                self.context_lens: context_lens[index],
                self.dropout_keep_prob: keep_prob,
            }
            yield feed_dict, len(index)
