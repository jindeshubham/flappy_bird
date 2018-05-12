import tensorflow as tf
import random
import numpy as np
import sys
import os
import cv2
from collections import deque
import time
sys.path.append("game/")
import wrapped_flappy_bird as game


class Q_network():
    def __init__(self, image_width, image_height,name='network'):
        self.image_width = image_width
        self.image_height = image_height
        self.name = name
        with tf.variable_scope(self.name) as scope:
            self.inputs = tf.placeholder(tf.float32, [None, self.image_width, self.image_height, 4], name='inputs')

            conv1 = tf.layers.conv2d(self.inputs, 32, 2, strides=1, padding='SAME', name='conv1')
            conv1 = tf.layers.max_pooling2d(conv1, 2, 1, padding='SAME')
            conv1 = tf.nn.relu(conv1)

            conv2 = tf.layers.conv2d(conv1, 64, 2, strides=1, padding='SAME', name='conv2')
            conv2 = tf.layers.max_pooling2d(conv2, 2, 1, padding='SAME')
            conv2 = tf.nn.relu(conv2)

            shape = conv2.get_shape().as_list()
            flat = tf.reshape(conv2, [-1, shape[1] * shape[2] * shape[3]])

            o_l1 = tf.nn.relu(tf.layers.dense(flat, 128))

            self.output = tf.layers.dense(o_l1, 2)

            self.model_vars = tf.trainable_variables()

# class Memory():
#     def __init__(self, max_size=3000):
#         self.buffer = deque(maxlen=max_size)
#
#     def add(self, experience):
#         self.buffer.append(experience)
#
#     def sample(self, batch_size):
#         idx = np.random(np.arange(len(self.buffer)),
#                         size=batch_size,
#                         replace=False
#                         )
#         return [self.buffer[ii] for ii in idx]


##Hyperparameters
class flappy():
    def __init__(self):
        self.learning_rate = 0.0001
        self.batch_size = 32
        self.memory_size = 10000
        self.train_episodes = 1000
        self.max_step = 200
        self.gamma = 0.99
        self.image_height = 80
        self.image_width = 80
        self.explore_start = 1
        self.explore_stop = 0.01
        self.decay_rate = 0.0001
        self.pre_train_steps = 100
        self.action_size = 2

    def pre_process(self, img):
        x_t = cv2.cvtColor(cv2.resize(img, (80, 80)), cv2.COLOR_BGR2GRAY)
        ret, x_t = cv2.threshold(x_t, 1, 255, cv2.THRESH_BINARY)

        return x_t

    def model(self):
        self.actions = tf.placeholder(tf.int32,[None,1],name='actions')
        one_hot_actions = tf.one_hot(tf.reshape(self.actions, [-1]), 2, dtype=tf.float32)
        self.targetQs = tf.placeholder(tf.float32, [None, 1], name='target_qs')
        self.reward  = tf.reduce_sum(self.qnetwork.output*tf.one_hot(tf.reshape(self.actions,[-1]),2,dtype=tf.float32),1,keep_dims=True)
        self.Q = tf.reduce_sum(tf.multiply(self.reward, one_hot_actions), axis=1)
        self.loss = tf.reduce_mean(tf.square(self.targetQs - self.Q))

        optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
        self.opt = optimizer.minimize(self.loss)


    def train(self):
        #saver = tf.train.Saver()
        reward_list = []
        with tf.Session() as sess:
            self.qnetwork = Q_network(80, 80,name='qnetwork')
            self.target_network = Q_network(80,80,name='targetnet')
            sess.run(tf.initialize_all_variables())
            # Pretrain the network a little. Add little blocks of memory
            self.model()
            game_state = game.GameState()
            action = np.zeros([2])
            memory = []
            temp_action = random.randint(0, 1)
            action[temp_action] = 1
            new_state, reward, done = game_state.frame_step(action)
            temp_img = self.pre_process(new_state)
            img_batch = [temp_img] * 4
            print (img_batch)

            step = 0
            for episode in range(1, self.train_episodes):
                total_reward = 0
                t = 0
                while True:
                    startime = time.clock()
                    print ("startime "+str(startime))
                    explore_p = self.explore_stop + (self.explore_start - self.explore_stop) * np.exp(
                        step * self.decay_rate)

                    if explore_p < np.random.rand():
                        temp_action = random.randint(0, 1)
                    else:
                        temp_q_values = sess.run([self.qnetwork.output], feed_dict={
                            self.qnetwork.inputs: np.reshape(np.stack(img_batch, axis=2), [-1, 80, 80, 4])})
                        temp_action = np.argmax(temp_q_values)
                    second_time = time.clock() - startime
                    print("Second time "+str(second_time))
                    action = np.zeros([2])
                    action[temp_action] = 1

                    new_state, reward, done = game_state.frame_step(action)
                    temp_img = self.pre_process(new_state)
                    total_reward += reward
                    new_img_batch = img_batch[1:]
                    new_img_batch.insert(3, temp_img)
                    memory.append(
                        (np.stack(img_batch, axis=2), temp_action, reward, np.stack(new_img_batch, axis=2), done))
                    img_batch.insert(len(img_batch), temp_img)
                    img_batch.pop(0)

                    if step > self.pre_train_steps:
                        random_batch = random.sample(memory, self.batch_size)
                        third_time = time.clock() - second_time
                        print ("third time " + str(third_time))
                        reward_hist = [m[2] for m in random_batch]
                        state_hist = [m[0] for m in random_batch]
                        action_hist = [m[1] for m in random_batch]
                        next_state_hist = [m[3] for m in random_batch]
                        forth_time = time.clock() - third_time
                        print ("Forth time "+str(forth_time))
                        temp_target_q = sess.run(self.target_network.output, feed_dict={
                            self.target_network.inputs: np.stack(next_state_hist)})
                        temp_target_q = np.amax(temp_target_q, 1)
                        temp_target_reward = reward_hist + self.gamma * temp_target_q
                        temp_target_reward = np.reshape(temp_target_reward, [self.batch_size, 1])
                        fifth_time =time.clock() - forth_time
                        print ("Fifth time "+str(fifth_time))
                        print("5")
                        _ = sess.run(self.opt, feed_dict={
                            self.qnetwork.inputs: np.stack(state_hist),
                            self.targetQs: temp_target_reward,
                            self.actions: np.reshape(np.stack(action_hist), [self.batch_size, 1])

                        })
                        print ("Sixth time "+str(time.clock()-fifth_time))
                        print("6")
                    if (done):
                        print("7")
                        break

                    step += 1

                print("Total rewards in episode " + str(episode) + " is " + str(
                        total_reward) + " total number of steps are " + str(step))


def main():
    flap = flappy()
    flap.train()


if __name__ == "__main__":
    main()
