import os
import metrics
from metrics import timeit

################ Replay Buffer for DDPG ######################
from collections import deque
import random


class ReplayBuffer(object):

  def __init__(self, buffer_size, random_seed=123):
    """
    The right side of the deque contains the most recent experiences 
    """
    self.buffer_size = buffer_size
    self.count = 0
    self.buffer = deque()
    random.seed(random_seed)

  def add(self, s, a, r, t, s2):
    experience = (s, a, r, t, s2)
    if self.count < self.buffer_size:
      self.buffer.append(experience)
      self.count += 1
    else:
      self.buffer.popleft()
      self.buffer.append(experience)

  def size(self):
    return self.count

  def sample_batch(self, batch_size):
    batch = []

    if self.count < batch_size:
      batch = random.sample(self.buffer, self.count)
    else:
      batch = random.sample(self.buffer, batch_size)

    s_batch = np.array([_[0] for _ in batch])
    a_batch = np.array([_[1] for _ in batch])
    r_batch = np.array([_[2] for _ in batch])
    t_batch = np.array([_[3] for _ in batch])
    s2_batch = np.array([_[4] for _ in batch])

    return s_batch, a_batch, r_batch, t_batch, s2_batch

  def clear(self):
    self.buffer.clear()
    self.count = 0


################ DDPG Algorithm ######################
import tensorflow as tf
import tflearn
import numpy as np

class ActorNetwork(object):

    def __init__(self, sess, actor_structure,state_dim, action_dim, action_bound, learning_rate, tau, batch_size):
        self.sess = sess
        self.actor_structure = actor_structure
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.action_bound = action_bound
        self.learning_rate = learning_rate
        self.tau = tau
        self.batch_size = batch_size

        # Actor Network
        self.inputs, self.out, self.scaled_out = self.create_actor_network()

        self.network_params = tf.trainable_variables()

        # Target Network
        self.target_inputs, self.target_out, self.target_scaled_out = self.create_actor_network()

        self.target_network_params = tf.trainable_variables()[
            len(self.network_params):]

        # Op for periodically updating target network with online network
        # weights
        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) +
                                                  tf.multiply(self.target_network_params[i], 1. - self.tau))
                for i in range(len(self.target_network_params))]

        # This gradient will be provided by the critic network
        self.action_gradient = tf.placeholder(tf.float32, [None, self.a_dim])

        # Combine the gradients here
        self.unnormalized_actor_gradients = tf.gradients(
            self.scaled_out, self.network_params, -self.action_gradient)
        self.actor_gradients = list(map(lambda x: tf.div(x, self.batch_size), self.unnormalized_actor_gradients))

        # Optimization Op
        self.optimize = tf.train.AdamOptimizer(self.learning_rate).\
            apply_gradients(zip(self.actor_gradients, self.network_params))

        self.num_trainable_vars = len(
            self.network_params) + len(self.target_network_params)

    def create_actor_network(self):
        inputs = tflearn.input_data(shape=[None, self.s_dim])
        net = inputs
        for layer_nueral_number in self.actor_structure:
            net = tflearn.fully_connected(net, layer_nueral_number)
            net = tflearn.layers.normalization.batch_normalization(net)
            net = tflearn.activations.relu(net)
        
        # Final layer weights are init to Uniform[-3e-3, 3e-3]
        w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
        out = tflearn.fully_connected(
            net, self.a_dim, activation='tanh', weights_init=w_init)
        # Scale output to -action_bound to action_bound
        scaled_out = tf.multiply(out, self.action_bound)
        return inputs, out, scaled_out

    def save_npz(self, session):
        params = dict((parameter.name, parameter.eval(session=session))
                      for parameter in self.network_params)
        np.savez("network.npz", **params)

    def train(self, inputs, a_gradient):
        self.sess.run(self.optimize, feed_dict={
            self.inputs: inputs,
            self.action_gradient: a_gradient
        })

    def predict(self, inputs):
        return self.sess.run(self.scaled_out, feed_dict={
            self.inputs: inputs
        })

    def predict_target(self, inputs):
        return self.sess.run(self.target_scaled_out, feed_dict={
            self.target_inputs: inputs
        })

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

    def get_num_trainable_vars(self):
        return self.num_trainable_vars


class CriticNetwork(object):
    """
    Input to the network is the state and action, output is Q(s,a).
    The action must be obtained from the output of the Actor network.

    """

    def __init__(self, sess, critic_structure, state_dim, action_dim, learning_rate, tau, gamma, num_actor_vars):
        self.sess = sess
        self.critic_structure = critic_structure
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.learning_rate = learning_rate
        self.tau = tau
        self.gamma = gamma

        # Create the critic network
        self.inputs, self.action, self.out = self.create_critic_network()
        self.network_params = tf.trainable_variables()[num_actor_vars:]

        # Target Network
        self.target_inputs, self.target_action, self.target_out = self.create_critic_network()
        self.target_network_params = tf.trainable_variables()[(len(self.network_params) + num_actor_vars):]

        # Op for periodically updating target network with online network
        # weights with regularization
        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) \
            + tf.multiply(self.target_network_params[i], 1. - self.tau))
                for i in range(len(self.target_network_params))]

        # Network target (y_i)
        self.predicted_q_value = tf.placeholder(tf.float32, [None, 1])

        # Define loss and optimization Op
        self.loss = tflearn.mean_square(self.predicted_q_value, self.out)
        self.optimize = tf.train.AdamOptimizer(
            self.learning_rate).minimize(self.loss)

        # Get the gradient of the net w.r.t. the action.
        # For each action in the minibatch (i.e., for each x in xs),
        # this will sum up the gradients of each critic output in the minibatch
        # w.r.t. that action. Each output is independent of all
        # actions except for one.
        self.action_grads = tf.gradients(self.out, self.action)

    def create_critic_network(self):
        inputs = tflearn.input_data(shape=[None, self.s_dim])
        action = tflearn.input_data(shape=[None, self.a_dim])
        net = inputs

        for layer_nueral_number in self.critic_structure[:-1]:
            net = tflearn.fully_connected(inputs, layer_nueral_number)
            net = tflearn.layers.normalization.batch_normalization(net)
            net = tflearn.activations.relu(net)

        # Add the action tensor in the 2nd hidden layer
        # Use two temp layers to get the corresponding weights and biases
        t1 = tflearn.fully_connected(net, self.critic_structure[-1])
        t2 = tflearn.fully_connected(action, self.critic_structure[-1])

        net = tflearn.activation(
            tf.matmul(net, t1.W) + tf.matmul(action, t2.W) + t2.b, activation='relu')

        # linear layer connected to 1 output representing Q(s,a)
        # Weights are init to Uniform[-3e-3, 3e-3]
        w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
        out = tflearn.fully_connected(net, 1, weights_init=w_init)
        return inputs, action, out

    def train(self, inputs, action, predicted_q_value):
        return self.sess.run([self.out, self.optimize], feed_dict={
            self.inputs: inputs,
            self.action: action,
            self.predicted_q_value: predicted_q_value
        })

    def predict(self, inputs, action):
        return self.sess.run(self.scaled_out, feed_dict={
            self.inputs: inputs,
            self.action: action
        })

    def predict_target(self, inputs, action):
        return self.sess.run(self.target_out, feed_dict={
            self.target_inputs: inputs,
            self.target_action: action
        })

    def action_gradients(self, inputs, actions):
        return self.sess.run(self.action_grads, feed_dict={
            self.inputs: inputs,
            self.action: actions
        })

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

# Taken from https://github.com/openai/baselines/blob/master/baselines/ddpg/noise.py, which is
# based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
class OrnsteinUhlenbeckActionNoise:
    def __init__(self, mu, sigma=0.3, theta=.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
                self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)

@timeit
def train(sess, env, args, actor, critic, actor_noise, restorer, replay_buffer=None):

    # Initialize target network weights
    actor.update_target_network()
    critic.update_target_network()

    # Initialize replay memory
    if replay_buffer is None:
      replay_buffer = ReplayBuffer(int(args['buffer_size']), int(args['random_seed']))

    # Needed to enable BatchNorm. 
    # This hurts the performance on Pendulum but could be useful in other environments.
    # tflearn.is_training(True)

    last_reward = env.bad_reward
    count = 0
    reward_list = []

    for i in range(int(args['max_episodes'])):
        s = env.reset()
        ep_reward = 0
        ep_ave_max_q = 0
        #temp_r = env.bad_reward

        for j in range(int(args['max_episode_len'])):
            # Added exploration noise
            a = actor.predict(np.reshape(s, (1, actor.s_dim))) + actor_noise()

            s2, r, terminal = env.step(a.reshape(actor.a_dim, 1))

            # if r > temp_r:
            #   temp_r = r
            replay_buffer.add(np.reshape(np.array(s), (actor.s_dim,)), np.reshape(np.array(a), (actor.a_dim,)), r,
                                terminal, np.reshape(np.array(s2), (actor.s_dim, )))

            # Keep adding experience to the memory until
            # there are at least minibatch size samples
            if replay_buffer.size() > int(args['minibatch_size']):
                s_batch, a_batch, r_batch, t_batch, s2_batch = \
                    replay_buffer.sample_batch(int(args['minibatch_size']))

                # Calculate targets
                target_q = critic.predict_target(
                    s2_batch, actor.predict_target(s2_batch))

                y_i = []
                for k in range(int(args['minibatch_size'])):
                    if t_batch[k]:
                        y_i.append(r_batch[k])
                    else:
                        y_i.append(r_batch[k] + critic.gamma * target_q[k])

                # Update the critic given the targets
                predicted_q_value, _ = critic.train(
                    s_batch, a_batch, np.reshape(y_i, (int(args['minibatch_size']), 1)))

                ep_ave_max_q += np.amax(predicted_q_value)

                # Update the actor policy using the sampled gradient
                a_outs = actor.predict(s_batch)
                grads = critic.action_gradients(s_batch, a_outs)
                actor.train(s_batch, grads[0])

                # Update target networks
                actor.update_target_network()
                critic.update_target_network()

            s = s2
            ep_reward += r

            if terminal:
              # print "Termainal at step", j
              if j < int(args['max_episode_len']):
                s = env.reset()
                continue

        count += 1

        reward_list.append(ep_reward)
        if count % 10 == 0:

            reward_mean = np.mean(reward_list)

            if reward_mean > last_reward:
                # print reward_mean
                # print env.bad_reward
                restorer.save(sess, args['model_path'])
                print 'sess has been stored to', args['model_path']
                last_reward = reward_mean
            
            count = 0
            del reward_list[:]

        print('| Reward: {:.4f} | Episode: {:d} | Qmax: {:.4f}'.format(float(ep_reward), i, (ep_ave_max_q / float(j+1))))
        # print "x\n", env.xk 
        # print "u\n", env.last_u 


    print 'min reward:', last_reward
    if last_reward == env.bad_reward:
        restorer.save(sess, args['model_path'])
    model_path = os.path.split(args['model_path'])[0]+'/'
    final_model = model_path+'final_model.chkp'
    restorer.save(sess, final_model)
    print 'sess has been saved to', final_model

                    
@timeit
def test(env, actor, args, actor_noise):
  fail_time = 0
  success_time = 0
  fail_list = []

  for ep in xrange(args['test_episodes']):
    s = env.reset()
    init_s = s
    print "----ep: {} ----".format(ep)
    for i in xrange(args['test_episodes_len']):
      a = actor.predict(np.reshape(np.array(s), (1, actor.s_dim))) #+ actor_noise()
      s, r, terminal = env.step(a.reshape(actor.a_dim, 1))

      if terminal:
        if i != args['test_episodes_len']-1:
          if np.abs(r) < env.terminal_err:
            success_time += 1
          else:
            fail_time += 1
            fail_list.append((init_s, s))
          break
      elif i == args['test_episodes_len']-1:
        success_time += 1
    
    print 'initial state:\n', init_s, '\nstate at terminal step:\n'.format(i), s, '\nlast action:\n', env.last_u
    print "----terminal step: {} ----".format(i)

  print 'Success: {}, Fail: {}'.format(success_time, fail_time)

  print '#############Fail List:###############'
  for (i, e) in fail_list:
    print 'initial state: \n{}\nend state: \n{}\n----'.format(i, e)


def DDPG(env, args, replay_buffer=None):
    sess = tf.Session()
    np.random.seed(int(args['random_seed']))
    tf.set_random_seed(int(args['random_seed']))

    state_dim = env.state_dim
    action_dim = env.action_dim
    assert (env.u_max == -env.u_min).all()
    action_bound = env.u_max[0]

    actor = ActorNetwork(sess, list(args['actor_structure']), state_dim, action_dim, action_bound,
                         float(args['actor_lr']), float(args['tau']),
                         int(args['minibatch_size']))

    critic = CriticNetwork(sess, list(args['critic_structure']), state_dim, action_dim,
                           float(args['critic_lr']), float(args['tau']),
                           float(args['gamma']),
                           actor.get_num_trainable_vars())

    sess.run(tf.global_variables_initializer())
    restorer = tf.train.Saver(tf.global_variables())

    if tf.train.checkpoint_exists(args['model_path']):
        restorer.restore(sess, args['model_path'])
        print 'sess has been restored from', args['model_path']
        actor.save_npz(sess)
    
    actor_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(action_dim))

    train(sess, env, args, actor, critic, actor_noise, restorer, replay_buffer)

    if args['enable_test']:
        test(env, actor, args, actor_noise)

    return actor
    
@timeit
def actor_boundary(env, actor, epsoides=1000, steps=100):
  max_boundary = np.zeros([env.state_dim, 1])
  min_boundary = np.zeros([env.state_dim, 1])

  for ep in xrange(epsoides):
    s = env.reset()
    max_boundary, min_boundary = metrics.find_boundary(s, max_boundary, min_boundary)
    for i in xrange(steps):
      a = actor.predict(np.reshape(np.array(s), (1, actor.s_dim))) #+ actor_noise()
      s, _, terminal = env.step(a.reshape(actor.a_dim, 1))
      max_boundary, min_boundary = metrics.find_boundary(s, max_boundary, min_boundary)
      if terminal:
        break
    
  print 'max_boundary:\n{}\nmin_boundary:\n{}'.format(max_boundary, min_boundary)

@timeit
def random_search_for_init_buffer(env, args, target, trance_number, rewardf, max_count=5000, terminal_err=1, repeat_time=100, buffer_size=1000000):
  action_list = []
  xk_list_batch = []
  action_list_batch = []
  for i in xrange(trance_number):
      env.reset(target)
      xk, r, terminal = env.observation()
      count = 0
      xk_list = [env.xk]
      r_list = [r]
      action_list = []
      while not terminal:
        # a = float(raw_input("action: "))
        # a = np.array([[a]])
        a = np.random.uniform(-1, 1, [env.action_dim, 1])
        xk = env.simulation(a)
        if rewardf is None:
          def rewardf(xk, u):
            pass
        # Bad Terminal
        if (((np.array(xk) < env.x_max)*(np.array(xk) > env.x_min)).all(axis=1).any()) or rewardf(xk, a) < r:
          count += 1
          if count == max_count-1:
            count = 0
            if len(action_list) != 0:
              env.xk = xk_list.pop()
              action_list.pop()
              r = r_list.pop()
            else: 
              env.reset()
              xk, r, terminal = env.observation()
              xk_list = [env.xk]
              r_list = [r]
          continue

        # Good Terminal
        if np.sum(np.abs(xk-target)) < terminal_err:
          print "process: {}/{}".format(i+1,trance_number)
          xk_list.append(xk)
          action_list.append(a)
          break

        xk, r, terminal = env.step(a)
        action_list.append(a)
        xk_list.append(xk)
        r_list.append(r)
        count = 0

      print "end state:\n", xk_list[-1], "\n-----"

      for _ in xrange(repeat_time):
        xk_list_batch.append(xk_list)
        action_list_batch.append(action_list)
      # model_path = os.path.split(args['model_path'])[0]+'/'
      # model_path = model_path+'batch.json'
      batch = zip([xk_l[0] for xk_l in xk_list_batch], action_list_batch)


  return generate_replay_buffer(env, batch, buffer_size)

def generate_replay_buffer(env, batch, buffer_size):
  replay_buffer = ReplayBuffer(buffer_size)
  for x0, action_list in batch:
    env.reset(x0)
    for u in action_list:
      x1 = env.xk
      _, r, terminal = env.step(u)
      replay_buffer.add(np.reshape(np.array(x1), (env.state_dim,)), np.reshape(np.array(u), (env.action_dim,)), r, terminal, np.reshape(np.array(env.xk), (env.state_dim, )))
  return replay_buffer

@timeit
def generate_replay_buffer_with_K(K, env, buffer_size, epsoides, steps):
  replay_buffer = ReplayBuffer(buffer_size)
  for i in xrange(epsoides):
    xk = env.reset()
    last_x = xk
    for j in xrange(steps):
      u = K.dot(xk)
      xk, r, terminal = env.step(u)
      last_x = xk
      replay_buffer.add(np.reshape(np.array(last_x), (env.state_dim,)), np.reshape(np.array(u), (env.action_dim,)), r, terminal, np.reshape(np.array(xk), (env.state_dim, )))

  return replay_buffer
