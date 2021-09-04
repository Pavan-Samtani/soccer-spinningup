import numpy as np
import os.path as osp
import tensorflow as tf
import gym
import time

from core import ReplayBuffer
from spinup.algos.tf1.td3 import core
from spinup.algos.tf1.td3.core import get_vars
from spinup.user_config import DEFAULT_DATA_DIR
from spinup.utils.logx import EpochLogger


def td3(env_fn, actor_critic=core.mlp_actor_critic, ac_kwargs=dict(), seed=None, 
        steps_per_epoch=10000, epochs=10000, replay_size=int(2e6), gamma=0.99, 
        polyak=0.995, pi_lr=1e-4, q_lr=1e-4, batch_size=256, start_steps=50000, 
        update_after=10000, update_every=50, act_noise=0.1, target_noise=0.1, 
        noise_clip=0.5, policy_delay=2, num_test_episodes=50, max_ep_len=300, 
        logger_kwargs=dict(), save_freq=1, sess=None, render=False, 
        test_env_fn=None):
    """
    Twin Delayed Deep Deterministic Policy Gradient (TD3)


    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.

        actor_critic: A function which takes in placeholder symbols 
            for state, ``x_ph``, and action, ``a_ph``, and returns the main 
            outputs from the agent's Tensorflow computation graph:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``pi``       (batch, act_dim)  | Deterministically computes actions
                                           | from policy given states.
            ``q1``       (batch,)          | Gives one estimate of Q* for 
                                           | states in ``x_ph`` and actions in
                                           | ``a_ph``.
            ``q2``       (batch,)          | Gives another estimate of Q* for 
                                           | states in ``x_ph`` and actions in
                                           | ``a_ph``.
            ``q1_pi``    (batch,)          | Gives the composition of ``q1`` and 
                                           | ``pi`` for states in ``x_ph``: 
                                           | q1(x, pi(x)).
            ===========  ================  ======================================

        ac_kwargs (dict): Any kwargs appropriate for the actor_critic 
            function you provided to TD3.

        seed (int): Seed for random number generators.

        steps_per_epoch (int): Number of steps of interaction (state-action pairs) 
            for the agent and the environment in each epoch.

        epochs (int): Number of epochs to run and train agent.

        replay_size (int): Maximum length of replay buffer.

        gamma (float): Discount factor. (Always between 0 and 1.)

        polyak (float): Interpolation factor in polyak averaging for target 
            networks. Target networks are updated towards main networks 
            according to:

            .. math:: \\theta_{\\text{targ}} \\leftarrow 
                \\rho \\theta_{\\text{targ}} + (1-\\rho) \\theta

            where :math:`\\rho` is polyak. (Always between 0 and 1, usually 
            close to 1.)

        pi_lr (float): Learning rate for policy.

        q_lr (float): Learning rate for Q-networks.

        batch_size (int): Minibatch size for SGD.

        start_steps (int): Number of steps for uniform-random action selection,
            before running real policy. Helps exploration.

        update_after (int): Number of env interactions to collect before
            starting to do gradient descent updates. Ensures replay buffer
            is full enough for useful updates.

        update_every (int): Number of env interactions that should elapse
            between gradient descent updates. Note: Regardless of how long 
            you wait between updates, the ratio of env steps to gradient steps 
            is locked to 1.
            
        act_noise (float): Stddev for Gaussian exploration noise added to 
            policy at training time. (At test time, no noise is added.)

        target_noise (float): Stddev for smoothing noise added to target 
            policy.

        noise_clip (float): Limit for absolute value of target policy 
            smoothing noise.

        policy_delay (int): Policy will only be updated once every 
            policy_delay times for each update of the Q-networks.

        num_test_episodes (int): Number of episodes to test the deterministic
            policy at the end of each epoch.

        max_ep_len (int): Maximum length of trajectory / episode / rollout.

        logger_kwargs (dict): Keyword args for EpochLogger.

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.

    """

    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())
    max_ep_ret = -1e6
    success_rate = 0

    if seed is not None:
        tf.set_random_seed(seed)
        np.random.seed(seed)

    env, test_env_fn = env_fn(), test_env_fn if test_env_fn is not None else env_fn
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # Action limit for clamping: critically, assumes all dimensions share the same bound!
    act_limit = env.action_space.high[0]

    # Share information about action space with policy architecture
    ac_kwargs['action_space'] = env.action_space
    num_players = env.num_players
    assert num_players == test_env_fn().num_players

    if sess is None:
        sess = tf.Session()

    # Inputs to computation graph
    x_ph, a_ph, x2_ph, r_ph, d_ph = core.placeholders(obs_dim, act_dim, obs_dim, None, None)

    # Main outputs from computation graph
    with tf.variable_scope('main'):
        pi, q1, q2, q1_pi = actor_critic(x_ph, a_ph, **ac_kwargs)
    
    # Target policy network
    with tf.variable_scope('target'):
        pi_targ, _, _, _  = actor_critic(x2_ph, a_ph, **ac_kwargs)
    
    # Target Q networks
    with tf.variable_scope('target', reuse=True):

        # Target policy smoothing, by adding clipped noise to target actions
        epsilon = tf.random_normal(tf.shape(pi_targ), stddev=target_noise)
        epsilon = tf.clip_by_value(epsilon, -noise_clip, noise_clip)
        a2 = pi_targ + epsilon
        a2 = tf.clip_by_value(a2, -act_limit, act_limit)

        # Target Q-values, using action from target policy
        _, q1_targ, q2_targ, _ = actor_critic(x2_ph, a2, **ac_kwargs)

    # Experience buffer
    replay_buffer = {i: ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size // num_players) for i in range(num_players)}

    # Count variables
    var_counts = tuple(core.count_vars(scope) for scope in ['main/pi', 'main/q1', 'main/q2', 'main'])
    print('\nNumber of parameters: \t pi: %d, \t q1: %d, \t q2: %d, \t total: %d\n'%var_counts)

    # Bellman backup for Q functions, using Clipped Double-Q targets
    min_q_targ = tf.minimum(q1_targ, q2_targ)
    backup = tf.stop_gradient(r_ph + gamma*(1-d_ph)*min_q_targ)

    # TD3 losses
    pi_loss = -tf.reduce_mean(q1_pi)
    q1_loss = tf.reduce_mean((q1-backup)**2)
    q2_loss = tf.reduce_mean((q2-backup)**2)
    q_loss = q1_loss + q2_loss

    # Separate train ops for pi, q
    pi_optimizer = tf.train.AdamOptimizer(learning_rate=pi_lr)
    q_optimizer = tf.train.AdamOptimizer(learning_rate=q_lr)

    train_pi_op = pi_optimizer.minimize(pi_loss, var_list=(get_vars('main/pi')))
    train_q_op = q_optimizer.minimize(q_loss, var_list=(get_vars('main/q')))

    sess.run(tf.global_variables_initializer())

    # Polyak averaging for target variables
    target_update = tf.group([tf.assign(v_targ, polyak*v_targ + (1-polyak)*v_main)
                              for v_main, v_targ in zip(get_vars('main'), get_vars('target'))])

    # Initializing targets to match main variables
    target_init = tf.group([tf.assign(v_targ, v_main)
                              for v_main, v_targ in zip(get_vars('main'), get_vars('target'))])


    sess.run(target_init)

    # Setup model saving
    logger.setup_tf_saver(sess, inputs={'x': x_ph, 'a': a_ph}, outputs={'pi': pi, 'q1': q1, 'q2': q2})

    def get_action(o, noise_scale, batch_size=1):
        a = sess.run(pi, feed_dict={x_ph: o.reshape(batch_size, -1)})
        a += noise_scale * np.random.randn(batch_size, act_dim)
        return [np.ravel(x) for x in np.split(np.clip(a, -act_limit, act_limit), batch_size, axis=0)]

    def test_agent():
        success_rate = 0
        avg_ret = np.zeros(num_players)
        test_env = test_env_fn() # maintain random seed
        for j in range(num_test_episodes):
            o = test_env.reset()
            d, ep_ret, ep_ret_sparse, ep_len = False, np.zeros(num_players), np.zeros(num_players), 0
            vel_to_ball = [[] for j in range(num_players)]
            while not(d or (ep_len == max_ep_len)):
                # Take deterministic actions at test time (noise_scale=0)
                o, r, d, _ = test_env.step(get_action(np.array(o), 0, num_players))
                if j == 0 and render:
                    test_env.render(close=d)
                ep_ret += np.array(r)
                ep_ret_sparse += np.array(test_env.timestep.reward)
                [vel_to_ball[j].append(test_env.timestep.observation[j]['stats_vel_to_ball']) for j in range(num_players)]
                ep_len += 1
            success_rate += (ep_len <= max_ep_len and test_env.timestep.reward[0] > 0) / num_test_episodes
            avg_ret += ep_ret / num_test_episodes

            ep_ret_dict = {}
            for i in range(num_players):
                ep_ret_dict[f"TestEpRet_P{i + 1}"] = ep_ret[i]
                ep_ret_dict[f"TestEpRetSparse_P{i + 1}"] = ep_ret_sparse[i]
                ep_ret_dict[f"TestEpStatsVelToBall_P{i + 1}"] = np.mean(vel_to_ball[i])

            logger.store(**ep_ret_dict, TestEpLen=ep_len)

        return success_rate, avg_ret

    start_time = time.time()
    o = env.reset()
    ep_ret, ep_len = np.zeros(num_players), 0
    total_steps = steps_per_epoch * epochs
    pkl_saved = False

    # Main loop: collect experience in env and update/log each epoch
    for t in range(total_steps):

        # Until start_steps have elapsed, randomly sample actions
        # from a uniform distribution for better exploration. Afterwards, 
        # use the learned policy (with some noise, via act_noise). 
        # Step the env

        if t > start_steps:
            a = get_action(np.array(o), act_noise, num_players)
        else:
            a = [env.action_space.sample() for obs in o]

        o2, r, d, _ = env.step(a)
        ep_ret += np.array(r)
        ep_len += 1

        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        d = False if ep_len==max_ep_len else d

        # Store experience to replay buffer
        [replay_buffer[j].store(o[j].copy(), a[j].copy(), r[j], o2[j].copy(), d) for j in range(num_players)]

        # Super critical, easy to overlook step: make sure to update 
        # most recent observation!
        o = o2

        # End of trajectory handling
        if d or (ep_len == max_ep_len):
            ep_ret_dict = {f"EpRet_P{i + 1}": ep_ret[i] for i in range(num_players)}
            logger.store(**ep_ret_dict, EpLen=ep_len)
            o, ep_ret, ep_len = env.reset(), np.zeros(num_players), 0

        # Update handling
        if t >= update_after and t % update_every == 0:
            for j in range(update_every):
                batch_dicts = [replay_buffer[j].sample_batch(batch_size // num_players) for j in range(num_players)]
                batch = {key: np.concatenate([batch_dicts[i][key] for i in range(num_players)], axis=0) for key in batch_dicts[0].keys()}
                feed_dict = {x_ph: batch['obs1'],
                             x2_ph: batch['obs2'],
                             a_ph: batch['acts'],
                             r_ph: batch['rews'],
                             d_ph: batch['done']
                            }
                q_step_ops = [q_loss, q1, q2, train_q_op]
                outs = sess.run(q_step_ops, feed_dict)
                logger.store(LossQ=outs[0], Q1Vals=outs[1], Q2Vals=outs[2])

                if j % policy_delay == 0:
                    # Delayed policy update
                    outs = sess.run([pi_loss, train_pi_op, target_update], feed_dict)
                    logger.store(LossPi=outs[0])

        # End of epoch wrap-up
        if (t+1) % steps_per_epoch == 0:
            epoch = (t+1) // steps_per_epoch

            # Test the performance of the deterministic version of the agent.
            act_suc_rate, act_avg_ret = test_agent()

            # Save model
            print(f"Best Success Rate: {int(success_rate * 100)}, Episode Return: {np.round(max_ep_ret, 2)}")
            print(f"Step Success Rate: {int(act_suc_rate * 100)}, Step Episode Return: {np.round(act_avg_ret, 2)}", end=". ")
            if ((epoch % save_freq == 0) or (epoch == epochs)) and (act_suc_rate >= success_rate):
                logger.save_state({'env': env}, None, not(pkl_saved))
                if not pkl_saved:
                    pkl_saved = True
                    tf.get_default_graph().finalize()
                success_rate = act_suc_rate
                max_ep_ret = act_avg_ret
                print("Saving model ...")
                print(f"New Best Success Rate: {int(success_rate * 100,)}, Average Episode Return: {np.round(max_ep_ret, 2)}")

            else:
                print("")

            if ((epoch % save_freq == 0) or (epoch == epochs)) and (act_suc_rate >= 0.8):
                logger.save_state({'env': env}, t)
                print("Saving model ...")

            # Log info about epoch
            if t >= update_after:
                logger.log_tabular('Epoch', epoch)

                for i in range(num_players):
                    logger.log_tabular(f'EpRet_P{i + 1}', with_min_and_max=True)
                    logger.log_tabular(f'TestEpRet_P{i + 1}', with_min_and_max=True)
                    logger.log_tabular(f'TestEpRetSparse_P{i + 1}', with_min_and_max=True)
                    logger.log_tabular(f'TestEpStatsVelToBall_P{i + 1}', with_min_and_max=True)

                logger.log_tabular('EpLen', with_min_and_max=True)
                logger.log_tabular('TestEpLen', with_min_and_max=True)
                logger.log_tabular('TotalEnvInteracts', t)
                logger.log_tabular('TestEpSuccessRate', act_suc_rate)
                logger.log_tabular('Q1Vals', with_min_and_max=True)
                logger.log_tabular('Q2Vals', with_min_and_max=True)
                logger.log_tabular('LossPi', average_only=True)
                logger.log_tabular('LossQ', average_only=True)
                logger.log_tabular('Time', time.time()-start_time)
                logger.dump_tabular()

if __name__ == '__main__':
    import argparse
    import dm_soccer2gym
    from math import ceil
    parser = argparse.ArgumentParser()
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument("--gpu", type=float, default=-1)
    parser.add_argument("--reward", type=str, default="sparse")
    parser.add_argument("--control_timestep", type=float, default=0.05)
    parser.add_argument("--time_limit", type=float, default=30.)
    args = parser.parse_args()

    from spinup.utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(f'td3_soccer_1vs0_{args.reward}_{args.control_timestep}', data_dir=osp.join(DEFAULT_DATA_DIR, "TD3/1vs0"), datestamp=True)
    
    if args.gpu > 0:
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu)

        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    else:
        sess = None
    
    td3(lambda : dm_soccer2gym.make('1vs0', task_kwargs={"rew_type": args.reward, "time_limit": args.time_limit, "disable_jump": True, 
        "dist_thresh": 0.03, 'control_timestep': args.control_timestep}), 
        test_env_fn=lambda : dm_soccer2gym.make('1vs0', task_kwargs={"rew_type": "dense", "time_limit": args.time_limit, "disable_jump": True, 
        "dist_thresh": 0.03, 'control_timestep': args.control_timestep, 'random_state': 69}),
        actor_critic=core.mlp_actor_critic_heads,
        gamma=args.gamma, epochs=args.epochs,
        logger_kwargs=logger_kwargs,
        sess=sess, max_ep_len=ceil(args.time_limit / args.control_timestep))

