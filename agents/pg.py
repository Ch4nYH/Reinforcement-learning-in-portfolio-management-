# -*- coding:utf-8 -*-
'''
@Author: Louis Liang
@time:2018/9/15 0:34
'''
import tensorflow as tf
import numpy as np
import os


class PG:
    def __init__(self, M, L, N, name, load_weights, trainable, number):
        # Initial buffer
        self.buffer = list()
        self.name = name
        self.learning_rate = 2e-2
        self.number = number
        # Build up models
        self.session = tf.compat.v1.Session()

        # Initial input shape
        self.M = M
        self.L = L
        self.N = N
        self.cost = 0.0025
        self.global_step = tf.Variable(0, trainable=False)

        self.state, self.w_previous, self.out = self.build_net()
        self.future_price = tf.compat.v1.placeholder(tf.float32, [None]+[self.M])
        self.pv_vector = tf.reduce_sum(self.out * self.future_price,
                                       reduction_indices=[1]) * self.pc()
        self.profit = tf.reduce_prod(self.pv_vector)
        self.loss = -tf.reduce_mean(tf.math.log(self.pv_vector))
        self.optimize = tf.compat.v1.train.AdamOptimizer(self.learning_rate)\
            .minimize(self.loss, global_step=self.global_step)

        # Initial saver
        self.saver = tf.compat.v1.train.Saver(max_to_keep=10)

        if load_weights:
            print("Loading Model")
            try:
                checkpoint = \
                    tf.train.get_checkpoint_state('./result/PG/{}/saved_network/'.format(self.number))
                if checkpoint and checkpoint.model_checkpoint_path:
                    tf.reset_default_graph()
                    self.saver.restore(self.session,
                                       checkpoint.model_checkpoint_path)
                    print("Successfully loaded:",
                          checkpoint.model_checkpoint_path)
                else:
                    print("Could not find old network weights")
                    self.session.run(tf.compat.v1.global_variables_initializer())

            except Exception:
                print("Could not find old network weights")
                self.session.run(tf.compat.v1.global_variables_initializer())
        else:
            self.session.run(tf.compat.v1.global_variables_initializer())

        if trainable:
            # Initial summary
            self.summary_writer = tf.compat.v1.summary.FileWriter('./result/PG/{}/'.format(self.number),
                                                        self.session.graph)
            self.summary_ops, self.summary_vars = self.build_summaries()

    # 建立 policy gradient 神经网络 (有改变)
    def build_net(self):
        state = tf.compat.v1.placeholder(tf.float32, shape=[None]+[self.M]+[self.L]+[self.N], name='market_situation')
        network = tf.compat.v1.layers.Conv2D(2, [1, 2],
                                   padding="valid",
                                   activation="relu")(state)
        width = network.get_shape()[2]
        network = tf.compat.v1.layers.Conv2D(48, [1, width],
                                   padding="valid",
                                   activation="relu",
                                   kernel_regularizer=tf.keras.regularizers.l2(9e-5))(network)
        w_previous = tf.compat.v1.placeholder(tf.float32, shape=[None, self.M])
        network = tf.concat([network, tf.reshape(w_previous,
                                                 [-1, self.M, 1, 1])], axis=3)
        network = tf.compat.v1.layers.Conv2D(1, [1, network.get_shape()[2]],
                                   padding="valid",
                                   activation="relu",
                                   kernel_regularizer=tf.keras.regularizers.l2(9e-5))(network)

        network = tf.layers.flatten(network)
        w_init = tf.random_uniform_initializer(-0.5, 0.5)
        out = tf.layers.dense(network, self.M,
                              activation=tf.nn.softmax, kernel_initializer=w_init)

        return state, w_previous, out

    def pc(self):
        return 1 - tf.reduce_sum(tf.abs(self.out[:, 1:] - self.w_previous[:, 1:]), axis=1) * self.cost

    def predict(self, s, a_previous):
        return self.session.run(self.out, feed_dict={self.state: s, self.w_previous: a_previous})

    # 存储回合 transition (有改变)
    def save_transition(self, s, p, action, action_previous):
        self.buffer.append((s, p, action, action_previous))

    # 学习更新参数 (有改变)
    def train(self):
        s, p, a, a_previous = self.get_buffer()
        profit, _ = self.session.run([self.profit, self.optimize],
                                     feed_dict={self.state: s,
                                                self.out: np.reshape(a, (-1, self.M)),
                                                self.future_price: np.reshape(p, (-1, self.M)),
                                                self.w_previous: np.reshape(a_previous, (-1, self.M))})

    def get_buffer(self):
        s = [data[0][0] for data in self.buffer]
        p = [data[1] for data in self.buffer]
        a = [data[2] for data in self.buffer]
        a_previous = [data[3] for data in self.buffer]
        return s, p, a, a_previous

    def reset_buffer(self):
        self.buffer = list()

    def save_model(self):
        path = './result/PG/{}/saved_network/'.format(self.number)
        if not os.path.exists(path):
            os.makedirs(path)
        self.saver.save(self.session, path + self.name, global_step=self.global_step)

    def write_summary(self, reward):
        summary_str = self.session.run(self.summary_ops, feed_dict={
            self.summary_vars[0]: reward,
        })
        self.summary_writer.add_summary(summary_str, self.session.run(self.global_step))

    def close(self):
        self.session.close()

    def build_summaries(self):
        self.reward = tf.Variable(0.)
        tf.compat.v1.summary.scalar('Reward', self.reward)
        summary_vars = [self.reward]
        summary_ops = tf.compat.v1.summary.merge_all()
        return summary_ops, summary_vars
