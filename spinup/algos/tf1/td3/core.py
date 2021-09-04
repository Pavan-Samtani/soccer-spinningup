import numpy as np
import tensorflow as tf

u_initializer = tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3)


class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for TD3 agents.
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs1_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.obs2_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size, act_dim], dtype=np.float32)
        self.rews_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs1_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(obs1=self.obs1_buf[idxs],
                    obs2=self.obs2_buf[idxs],
                    acts=self.acts_buf[idxs],
                    rews=self.rews_buf[idxs],
                    done=self.done_buf[idxs])


def placeholder(dim=None):
    return tf.placeholder(dtype=tf.float32, shape=(None,dim) if dim else (None,))

def placeholders(*args):
    return [placeholder(dim) for dim in args]

def mlp(x, hidden_sizes=(32,), activation=tf.tanh, output_activation=None, hl_layer_init=None, 
        last_layer_init=None):
    for h in hidden_sizes[:-1]:
        x = tf.layers.dense(x, units=h, activation=activation, kernel_initializer=hl_layer_init)
    return tf.layers.dense(x, units=hidden_sizes[-1], activation=output_activation, 
                           kernel_initializer=last_layer_init)

def get_vars(scope, ex_scope=""):
    test = lambda x, y: x in y if len(x) else False
    return [x for x in tf.global_variables() if scope in x.name and not(test(ex_scope, x.name))]

def count_vars(scope, ex_scope=""):
    v = get_vars(scope, ex_scope)
    return sum([np.prod(var.shape.as_list()) for var in v])

"""
Actor-Critics
"""
def mlp_actor_critic(x, a, hidden_sizes=(300,400), activation=tf.nn.relu, 
                     output_activation=tf.tanh, action_space=None):
    act_dim = a.shape.as_list()[-1]
    act_limit = action_space.high[0]
    with tf.variable_scope('pi'):
        pi = act_limit * mlp(x, list(hidden_sizes)+[act_dim], activation, output_activation)
    with tf.variable_scope('q1'):
        q1 = tf.squeeze(mlp(tf.concat([x,a], axis=-1), list(hidden_sizes)+[1], activation, None), axis=1)
    with tf.variable_scope('q2'):
        q2 = tf.squeeze(mlp(tf.concat([x,a], axis=-1), list(hidden_sizes)+[1], activation, None), axis=1)
    with tf.variable_scope('q1', reuse=True):
        q1_pi = tf.squeeze(mlp(tf.concat([x,pi], axis=-1), list(hidden_sizes)+[1], activation, None), axis=1)
    return pi, q1, q2, q1_pi


def mlp_actor_critic_heads(x, a, hidden_sizes={"head":(32, 64), "concat":(256, 256)}, activation=tf.nn.leaky_relu, 
                     output_activation=tf.tanh, action_space=None):
                     
    act_dim = a.shape.as_list()[-1]
    x_dim = x.shape.as_list()[-1]
    
    assert x_dim >= 18
    assert (x_dim - 18) % 6 == 0
    
    num_other_players = (x_dim - 18) // 6
    act_limit = action_space.high[0]
    
    with tf.variable_scope('pi'):
        heads_own  = [mlp(x[:, 2 * i:2 * (i + 1)], list(hidden_sizes["head"]), activation, activation) for i in range(9)]
        heads_other = [mlp(x[:, 18 + 6 * i:18 + 6 * (i + 1)], list(hidden_sizes["head"]), activation, activation) for i in range(num_other_players)]
        joint_head = mlp(tf.concat(heads_own + heads_other, axis=-1), list(hidden_sizes["concat"]), activation, activation)
        pi = act_limit * mlp(joint_head, [act_dim], activation, output_activation, last_layer_init=u_initializer)
        
    with tf.variable_scope('q1'):
        heads_own  = [mlp(tf.concat([x[:, 2 * i:2 * (i + 1)], a], axis=-1), list(hidden_sizes["head"]), activation, activation) for i in range(9)]
        heads_other = [mlp(tf.concat([x[:, 18 + 6 * i:18 + 6 * (i + 1)], a], axis=-1), list(hidden_sizes["head"]), activation, activation) for i in range(num_other_players)]
        q1 = tf.squeeze(mlp(tf.concat(heads_own + heads_other, axis=-1), list(hidden_sizes["concat"])+[1], activation, None), axis=1)
        
    with tf.variable_scope('q2'):
        heads_own  = [mlp(tf.concat([x[:, 2 * i:2 * (i + 1)], a], axis=-1), list(hidden_sizes["head"]), activation, activation) for i in range(9)]
        heads_other = [mlp(tf.concat([x[:, 18 + 6 * i:18 + 6 * (i + 1)], a], axis=-1), list(hidden_sizes["head"]), activation, activation) for i in range(num_other_players)]
        q2 = tf.squeeze(mlp(tf.concat(heads_own + heads_other, axis=-1), list(hidden_sizes["concat"])+[1], activation, None), axis=1)
        
    with tf.variable_scope('q1', reuse=True):
        heads_own  = [mlp(tf.concat([x[:, 2 * i:2 * (i + 1)], pi], axis=-1), list(hidden_sizes["head"]), activation, activation) for i in range(9)]
        heads_other = [mlp(tf.concat([x[:, 18 + 6 * i:18 + 6 * (i + 1)], pi], axis=-1), list(hidden_sizes["head"]), activation, activation) for i in range(num_other_players)]
        q1_pi = tf.squeeze(mlp(tf.concat(heads_own + heads_other, axis=-1), list(hidden_sizes["concat"])+[1], activation, None), axis=1)
        
    return pi, q1, q2, q1_pi


def mlp_actor_critic_heads_v2(x, a, hidden_sizes={"head":(32, 64), "concat":(256, 256)}, activation=tf.nn.leaky_relu, 
                     output_activation=tf.tanh, action_space=None, num_teammates=1):
                     
    act_dim = a.shape.as_list()[-1]
    x_dim = x.shape.as_list()[-1]
    
    assert x_dim >= 18
    assert (x_dim - 18) % 6 == 0
    
    num_other_players = (x_dim - 18) // 6
    assert num_teammates <= num_other_players
    num_opponents = num_other_players - num_teammates
    act_limit = action_space.high[0]
    
    with tf.variable_scope('pi'):
        heads_own  = [mlp(x[:, 2 * i:2 * (i + 1)], list(hidden_sizes["head"]), activation, activation) for i in range(9)]
        heads_op = mlp(tf.reshape(x[:, 18:18 + 6 * num_opponents], [-1, 6]), list(hidden_sizes["head"]), activation, activation)
        heads_op = tf.reshape(heads_op, [-1, num_opponents, hidden_sizes["head"][-1]])
        heads_op = [tf.reduce_min(heads_op, axis=1), tf.reduce_max(heads_op, axis=1)]
        heads_tm = [mlp(x[:, 18 + 6 * (i + num_opponents):18 + 6 * (num_opponents + i + 1)], list(hidden_sizes["head"]), activation, activation) \
            for i in range(num_teammates)]
        joint_head = mlp(tf.concat(heads_own + heads_op + heads_tm, axis=-1), list(hidden_sizes["concat"]), activation, activation)
        pi = act_limit * mlp(joint_head, [act_dim], activation, output_activation, last_layer_init=u_initializer)
        
    with tf.variable_scope('q1'):
        heads_own  = [mlp(tf.concat([x[:, 2 * i:2 * (i + 1)], a], axis=-1), list(hidden_sizes["head"]), activation, activation) for i in range(9)]
        heads_op = mlp(tf.reshape(x[:, 18:18 + 6 * num_opponents], [-1, 6]), list(hidden_sizes["head"]), activation, activation)
        heads_op = tf.reshape(heads_op, [-1, num_opponents, hidden_sizes["head"][-1]])
        heads_op = [tf.reduce_min(heads_op, axis=1), tf.reduce_max(heads_op, axis=1)]
        heads_tm = [mlp(x[:, 18 + 6 * (i + num_opponents):18 + 6 * (num_opponents + i + 1)], list(hidden_sizes["head"]), activation, activation) \
            for i in range(num_teammates)]
        q1 = tf.squeeze(mlp(tf.concat(heads_own + heads_op + heads_tm, axis=-1), list(hidden_sizes["concat"])+[1], activation, None), axis=1)
        
    with tf.variable_scope('q2'):
        heads_own  = [mlp(tf.concat([x[:, 2 * i:2 * (i + 1)], a], axis=-1), list(hidden_sizes["head"]), activation, activation) for i in range(9)]
        heads_op = mlp(tf.reshape(x[:, 18:18 + 6 * num_opponents], [-1, 6]), list(hidden_sizes["head"]), activation, activation)
        heads_op = tf.reshape(heads_op, [-1, num_opponents, hidden_sizes["head"][-1]])
        heads_op = [tf.reduce_min(heads_op, axis=1), tf.reduce_max(heads_op, axis=1)]
        heads_tm = [mlp(x[:, 18 + 6 * (i + num_opponents):18 + 6 * (num_opponents + i + 1)], list(hidden_sizes["head"]), activation, activation) \
            for i in range(num_teammates)]
        q2 = tf.squeeze(mlp(tf.concat(heads_own + heads_op + heads_tm, axis=-1), list(hidden_sizes["concat"])+[1], activation, None), axis=1)
        
    with tf.variable_scope('q1', reuse=True):
        heads_own  = [mlp(tf.concat([x[:, 2 * i:2 * (i + 1)], pi], axis=-1), list(hidden_sizes["head"]), activation, activation) for i in range(9)]
        heads_op = mlp(tf.reshape(x[:, 18:18 + 6 * num_opponents], [-1, 6]), list(hidden_sizes["head"]), activation, activation)
        heads_op = tf.reshape(heads_op, [-1, num_opponents, hidden_sizes["head"][-1]])
        heads_op = [tf.reduce_min(heads_op, axis=1), tf.reduce_max(heads_op, axis=1)]
        heads_tm = [mlp(x[:, 18 + 6 * (i + num_opponents):18 + 6 * (num_opponents + i + 1)], list(hidden_sizes["head"]), activation, activation) \
            for i in range(num_teammates)]
        q1_pi = tf.squeeze(mlp(tf.concat(heads_own + heads_op + heads_tm, axis=-1), list(hidden_sizes["concat"])+[1], activation, None), axis=1)
        
    return pi, q1, q2, q1_pi
