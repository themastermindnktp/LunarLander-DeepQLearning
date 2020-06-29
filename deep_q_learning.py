# Created by khanhdh on 6/9/20
import logging

logger = logging.getLogger('main')

import tensorflow
import numpy


class DeepQNetwork:
    def __init__(
            self,
            lr,                    # learn rate
            n_actions,             # number of actions
            name,                  # name
            input_dims,            # input dimensions
            fc1_dims=256,          # fully connected layer 1 dimensions
            fc2_dims=256,          # fully connected layer 2 dimensions
            chkpt_dir='tmp/dqn'    # checkpoint directory
    ):
        self.lr = lr
        self.n_actions = n_actions
        self.name = name
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.chkpt_dir = chkpt_dir
        self.input_dims = input_dims
        self.checkpoint_file = f"{chkpt_dir}/deepqnet_{name}.ckpt"

        self.session = tensorflow.Session()
        self.build_network()
        self.session.run(tensorflow.global_variables_initializer())
        self.saver = tensorflow.train.Saver()
        self.params = tensorflow.get_collection(
            tensorflow.GraphKeys.TRAINABLE_VARIABLES,
            scope=self.name
        )

    def build_network(self):
        with tensorflow.variable_scope(self.name):
            self.input = tensorflow.placeholder(
                tensorflow.float32,
                shape=[None, *self.input_dims],
                name='inputs'
            )
            self.actions = tensorflow.placeholder(
                tensorflow.float32,
                shape=[None, self.n_actions],
                name='actions_taken'
            )
            self.q_target = tensorflow.placeholder(
                tensorflow.float32,
                shape=[None, self.n_actions],
                name='q_value'
            )

            flat = tensorflow.layers.flatten(self.input)
            dense1 = tensorflow.layers.dense(
                flat,
                units=self.fc1_dims,
                activation=tensorflow.nn.relu
            )
            dense2 = tensorflow.layers.dense(
                dense1,
                units=self.fc2_dims,
                activation=tensorflow.nn.relu
            )

            self.q_value = tensorflow.layers.dense(dense2, self.n_actions)
            self.loss = tensorflow.reduce_mean(tensorflow.square(self.q_value - self.q_target))
            self.train_op = tensorflow.train.AdamOptimizer(self.lr).minimize(self.loss)

    def load_checkpoint(self):
        print('loading checkpoint...')
        self.saver.restore(self.session, self.checkpoint_file)

    def save_checkpoint(self):
        print('saving checkpoint...')
        self.saver.save(self.session, self.checkpoint_file)


class Agent:
    def __init__(
            self, name, alpha, gamma, mem_size, n_actions, epsilon, batch_size,
            n_games, input_dims=(210, 160, 4), epsilon_dec=0.996,
            epsilon_min=0.01, q_eval_dir='tmp/q_eval'
    ):
        self.action_space = list(range(n_actions))
        self.n_actions = n_actions
        self.n_games = n_games
        self.gamma = gamma
        self.mem_size = mem_size
        self.mem_cnt = 0
        self.epsilon = epsilon
        self.epsilon_dec = epsilon_dec
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.q_eval = DeepQNetwork(
            alpha, n_actions,
            name='q_val',
            input_dims=input_dims,
            chkpt_dir=q_eval_dir
        )
        self.state_memory = numpy.zeros((self.mem_size, *input_dims))
        self.new_state_memory = numpy.zeros((self.mem_size, *input_dims))
        self.action_memory = numpy.zeros((self.mem_size, self.n_actions), dtype=numpy.int8)
        self.reward_memory = numpy.zeros(self.mem_size)
        self.terminal_memory = numpy.zeros(self.mem_size, dtype=numpy.int8)

    def store_transition(self, state, action, reward, new_state, terminal):
        index = self.mem_cnt % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = new_state
        self.reward_memory[index] = reward
        actions = numpy.zeros(self.n_actions)
        actions[action] = 1.0
        self.action_memory[index] = actions
        self.terminal_memory[index] = 1 - terminal
        self.mem_cnt += 1

    def choose_action(self, state, randomize=True):
        state = state[numpy.newaxis, :]
        rand = numpy.random.random()
        if randomize and rand < self.epsilon:
            action = numpy.random.choice(self.action_space)
        else:
            actions = self.q_eval.session.run(
                self.q_eval.q_value,
                feed_dict={self.q_eval.input: state}
            )
            action = numpy.argmax(actions)
        return action

    def learn(self):
        if self.mem_cnt > self.batch_size:
            max_mem = min(self.mem_cnt, self.mem_size)
            batch = numpy.random.choice(max_mem, self.batch_size)

            state_batch = self.state_memory[batch]
            action_batch = self.action_memory[batch]
            action_values = numpy.array(self.action_space, dtype=numpy.int8)
            action_indices = numpy.dot(action_batch, action_values)
            reward_batch = self.reward_memory[batch]
            new_state_batch = self.new_state_memory[batch]
            terminal_batch = self.terminal_memory[batch]

            q_eval = self.q_eval.session.run(
                self.q_eval.q_value,
                feed_dict={self.q_eval.input: state_batch}
            )

            q_next = self.q_eval.session.run(
                self.q_eval.q_value,
                feed_dict={self.q_eval.input: new_state_batch}
            )

            q_target = q_eval.copy()

            batch_index = numpy.arange(self.batch_size, dtype=numpy.int32)
            q_target[batch_index, action_indices] = reward_batch + self.gamma*numpy.max(q_next, axis=1)*terminal_batch

            self.q_eval.session.run(
                self.q_eval.train_op,
                feed_dict={
                    self.q_eval.input: state_batch,
                    self.q_eval.actions: action_batch,
                    self.q_eval.q_target: q_target
                }
            )

            # self.epsilon = self.epsilon*self.epsilon_dec if self.epsilon > self.epsilon_min else self.epsilon_min
            self.epsilon = max(self.epsilon*self.epsilon_dec, self.epsilon_min)

    def save_models(self):
        self.q_eval.save_checkpoint()

    def load_models(self):
        self.q_eval.load_checkpoint()
