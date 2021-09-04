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
from spinup.utils.test_policy import load_policy_and_env


def td3(env_fn, actor_critic=core.mlp_actor_critic, ac_kwargs=dict(), seed=None, 
        steps_per_epoch=10000, epochs=10000, replay_size=int(2.5e6), gamma=0.99, 
        polyak=0.995, pi_lr=1e-4, q_lr=1e-4, batch_size=256, start_steps=10000, 
        update_after=10000, update_every=50, act_noise=0.1, target_noise=0.1, 
        noise_clip=0.5, policy_delay=2, num_test_episodes=50, max_ep_len=900, 
        logger_kwargs=dict(), save_freq=1, sess=None, load_1vs1="", num=0,
        render=False, test_env_fn=None, use_es=True):
        
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
    assert num_players == 4
    assert num_players == test_env_fn().num_players
    
    # Define indexes to use based on usage of experience sharing
    es_1_idxs = [0, 1, 2] if use_es else [2]
    es_2_idxs = [0, 1, 3] if use_es else [3]
    es_rb_idxs = [0, 1, 2, 3] if use_es else [2, 3]

    if sess is None:
        sess = tf.Session()

    # Inputs to computation graph
    x_ph, a_ph, x2_ph, r_ph, d_ph = core.placeholders(obs_dim, act_dim, obs_dim, None, None)

    # Main outputs from computation graph
    with tf.variable_scope('main'):
        with tf.variable_scope('player_1'):
            pi_1, q1_1, q2_1, q1_pi_1 = actor_critic(x_ph, a_ph, **ac_kwargs)
        with tf.variable_scope('player_2'):
            pi_2, q1_2, q2_2, q1_pi_2 = actor_critic(x_ph, a_ph, **ac_kwargs)
    
    # Target policy network
    with tf.variable_scope('target'):
        with tf.variable_scope('player_1'):
            pi_targ_1, _, _, _  = actor_critic(x2_ph, a_ph, **ac_kwargs)
        with tf.variable_scope('player_2'):
            pi_targ_2, _, _, _  = actor_critic(x2_ph, a_ph, **ac_kwargs)
    
    # Target Q networks
    with tf.variable_scope('target', reuse=True):

        # Target policy smoothing, by adding clipped noise to target actions
        epsilon_1 = tf.random_normal(tf.shape(pi_targ_1), stddev=target_noise)
        epsilon_1 = tf.clip_by_value(epsilon_1, -noise_clip, noise_clip)
        a2_1 = pi_targ_1 + epsilon_1
        a2_1 = tf.clip_by_value(a2_1, -act_limit, act_limit)
        
        epsilon_2 = tf.random_normal(tf.shape(pi_targ_2), stddev=target_noise)
        epsilon_2 = tf.clip_by_value(epsilon_2, -noise_clip, noise_clip)
        a2_2 = pi_targ_2 + epsilon_2
        a2_2 = tf.clip_by_value(a2_2, -act_limit, act_limit)

        # Target Q-values, using action from target policy
        with tf.variable_scope('player_1'):
            _, q1_targ_1, q2_targ_1, _ = actor_critic(x2_ph, a2_1, **ac_kwargs)
        with tf.variable_scope('player_2'):
            _, q1_targ_2, q2_targ_2, _ = actor_critic(x2_ph, a2_2, **ac_kwargs)
            

    # Experience buffer
    replay_buffer = {i: ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size) for i in es_rb_idxs}

    # Count variables
    var_counts = tuple(2 * core.count_vars(scope) for scope in ['main/player_1/pi', 'main/player_1/q1', 
                                                                'main/player_1/q2', 'main/player_1/'])
    print('\nNumber of parameters: \t pi: %d, \t q1: %d, \t q2: %d, \t total: %d\n'%var_counts)

    # Bellman backup for Q functions, using Clipped Double-Q targets
    min_q_targ_1 = tf.minimum(q1_targ_1, q2_targ_1)
    min_q_targ_2 = tf.minimum(q1_targ_2, q2_targ_2)
    backup_1 = tf.stop_gradient(r_ph + gamma * (1 - d_ph) * min_q_targ_1)
    backup_2 = tf.stop_gradient(r_ph + gamma * (1 - d_ph) * min_q_targ_2)

    # TD3 losses
    pi_loss_1 = -tf.reduce_mean(q1_pi_1)
    q1_loss_1 = tf.reduce_mean((q1_1 - backup_1) ** 2)
    q2_loss_1 = tf.reduce_mean((q2_1 - backup_1) ** 2)
    q_loss_1 = q1_loss_1 + q2_loss_1
    
    pi_loss_2 = -tf.reduce_mean(q1_pi_2)
    q1_loss_2 = tf.reduce_mean((q1_2 - backup_2) ** 2)
    q2_loss_2 = tf.reduce_mean((q2_2 - backup_2) ** 2)
    q_loss_2 = q1_loss_2 + q2_loss_2

    # Separate train ops for pi, q
    pi_optimizer_1 = tf.train.AdamOptimizer(learning_rate=pi_lr)
    q_optimizer_1 = tf.train.AdamOptimizer(learning_rate=q_lr)
    
    pi_optimizer_2 = tf.train.AdamOptimizer(learning_rate=pi_lr)
    q_optimizer_2 = tf.train.AdamOptimizer(learning_rate=q_lr)

    train_pi_op_1 = pi_optimizer_1.minimize(pi_loss_1, var_list=(get_vars('main/player_1/pi')))
    train_pi_op_2 = pi_optimizer_2.minimize(pi_loss_2, var_list=(get_vars('main/player_2/pi')))
    
    train_q_op_1 = q_optimizer_1.minimize(q_loss_1, var_list=(get_vars('main/player_1/q')))
    train_q_op_2 = q_optimizer_2.minimize(q_loss_2, var_list=(get_vars('main/player_2/q')))

    sess.run(tf.global_variables_initializer())
    
    assert len(load_1vs1) == 2
    assert 2 == len(num)
    
    g1 = tf.Graph()
    with g1.as_default():
        __, _get_1v1_action_p1 = load_policy_and_env(osp.join(DEFAULT_DATA_DIR, load_1vs1[0]), num[0], sess=None)
    
    g2 = tf.Graph()
    with g2.as_default():
        __, _get_1v1_action_p2 = load_policy_and_env(osp.join(DEFAULT_DATA_DIR, load_1vs1[1]), num[1], sess=None)
            
    get_1v1_action_1 = lambda x: [_get_1v1_action_p1(x)]
    get_1v1_action_2 = lambda x: [_get_1v1_action_p2(x)]

    # Polyak averaging for target variables
    target_update_1 = tf.group([tf.assign(v_targ, polyak*v_targ + (1-polyak)*v_main)
                                for v_main, v_targ in zip(get_vars('main/player_1'), get_vars('target/player_1'))])
                                
    target_update_2 = tf.group([tf.assign(v_targ, polyak*v_targ + (1-polyak)*v_main)
                                for v_main, v_targ in zip(get_vars('main/player_2'), get_vars('target/player_2'))])

    # Initializing targets to match main variables
    target_init_1 = tf.group([tf.assign(v_targ, v_main)
                                for v_main, v_targ in zip(get_vars('main/player_1'), get_vars('target/player_1'))])
                                
    target_init_2 = tf.group([tf.assign(v_targ, v_main)
                                for v_main, v_targ in zip(get_vars('main/player_2'), get_vars('target/player_2'))])

    sess.run([target_init_1, target_init_2])

    # Setup model saving
    logger.setup_tf_saver(sess, inputs={'x': x_ph, 'a': a_ph}, outputs={'pi_1': pi_1, 'q1_1': q1_1, 'q2_1': q2_1,
                                                                        'pi_2': pi_2, 'q1_2': q1_2, 'q2_2': q2_2})

    def get_action(o, noise_scale, batch_size=1):
        a_1 = sess.run(pi_1, feed_dict={x_ph: o[::2].reshape(batch_size, -1)})
        a_2 = sess.run(pi_2, feed_dict={x_ph: o[1::2].reshape(batch_size, -1)})
        a = np.zeros((a_1.shape[0] + a_2.shape[0], a_1.shape[1]))
        a[::2] = a_1
        a[1::2] = a_2
        a += noise_scale * np.random.randn(batch_size, act_dim)
        return [np.ravel(x) for x in np.split(np.clip(a, -act_limit, act_limit), 2 * batch_size, axis=0)]

    def test_agent():
        success_rate = 0
        avg_ret = np.zeros(num_players)
        test_env = test_env_fn()
        max_ep_len = test_env.time_limit // test_env.control_timestep
        for j in range(num_test_episodes):
            o = test_env.reset()
            d, ep_ret, ep_ret_sparse, ep_len = False, np.zeros(num_players), np.zeros(num_players), 0
            
            vel_to_ball = []
            spread_out = []
            intercepted = []
            intercepted_5 = []
            intercepted_10 = []
            intercepted_15 = []
            received = []
            received_5 = []
            received_10 = []
            received_15 = []
            received_p = []
            received_p_5 = []
            received_p_10 = []
            received_p_15 = []
            
            for k in range(num_players):
            
                vel_to_ball.append([])
                spread_out.append([])
                intercepted.append([])
                intercepted_5.append([])
                intercepted_10.append([])
                intercepted_15.append([])
                received.append([])
                received_5.append([])
                received_10.append([])
                received_15.append([])
                received_p.append([])
                received_p_5.append([])
                received_p_10.append([])
                received_p_15.append([])
            
            while not(d or (ep_len == max_ep_len)):
                # Take deterministic actions at test time (noise_scale=0)
                if j % 2 == 0:
                    act_1 = get_1v1_action_1(o[0][np.r_[0:18, 18:24]]) + get_1v1_action_2(o[1][np.r_[0:18, 24:30]])
                else:
                    act_1 = get_1v1_action_2(o[0][np.r_[0:18, 18:24]]) + get_1v1_action_1(o[1][np.r_[0:18, 24:30]])
                    
                a = act_1 + get_action(np.array(o[2:]), 0, (num_players - 2) // 2)

                o, r, d, _ = test_env.step(a)
                if j == 0 and render:
                    test_env.render()
                
                for k in range(num_players):
                
                    test_obs = test_env.timestep.observation[k]
                    _switch_k = (2 - k - 1) if ((k < 2) and (j % 2 == 1)) else k
                    
                    ep_ret[_switch_k] += r[k]
                    ep_ret_sparse[_switch_k] += test_env.timestep.reward[k]
                    vel_to_ball[_switch_k].append(test_obs['stats_vel_to_ball'])
                    spread_out[_switch_k].append(test_obs['stats_teammate_spread_out'])
                    intercepted[_switch_k].append(test_obs['stats_opponent_intercepted_ball'])
                    intercepted_5[_switch_k].append(test_obs['stats_opponent_intercepted_ball_5m'])
                    intercepted_10[_switch_k].append(test_obs['stats_opponent_intercepted_ball_10m'])
                    intercepted_15[_switch_k].append(test_obs['stats_opponent_intercepted_ball_15m'])
                    received[_switch_k].append(test_obs['stats_i_received_ball'])
                    received_5[_switch_k].append(test_obs['stats_i_received_ball_5m'])
                    received_10[_switch_k].append(test_obs['stats_i_received_ball_10m'])
                    received_15[_switch_k].append(test_obs['stats_i_received_ball_15m'])
                    received_p[_switch_k].append(test_obs['stats_i_received_pass'])
                    received_p_5[_switch_k].append(test_obs['stats_i_received_pass_5m'])
                    received_p_10[_switch_k].append(test_obs['stats_i_received_pass_10m'])
                    received_p_15[_switch_k].append(test_obs['stats_i_received_pass_15m'])
                    
                ep_len += 1
            success_rate += (ep_len <= max_ep_len and test_env.timestep.reward[0] < 0) / num_test_episodes
            avg_ret += ep_ret / num_test_episodes

            ep_ret_dict = {}
            for i in range(num_players):
                ep_ret_dict[f"TestEpRet_P{i + 1}"] = ep_ret[i]
                ep_ret_dict[f"TestEpRetSparse_P{i + 1}"] = ep_ret_sparse[i]
                ep_ret_dict[f"TestEpStatsVelToBall_P{i + 1}"] = np.mean(vel_to_ball[i])
                ep_ret_dict[f"TestEpStatsTeamSpreadOut_P{i + 1}"] = np.mean(spread_out[i])
                ep_ret_dict[f"TestEpStatsOpIntercepted_P{i + 1}"] = np.mean(intercepted[i])
                ep_ret_dict[f"TestEpStatsOpIntercepted_5m_P{i + 1}"] = np.mean(intercepted_5[i])
                ep_ret_dict[f"TestEpStatsOpIntercepted_10m_P{i + 1}"] = np.mean(intercepted_10[i])
                ep_ret_dict[f"TestEpStatsOpIntercepted_15m_P{i + 1}"] = np.mean(intercepted_15[i])
                ep_ret_dict[f"TestEpStatsIReceived_P{i + 1}"] = np.mean(received[i])
                ep_ret_dict[f"TestEpStatsIReceived_5m_P{i + 1}"] = np.mean(received_5[i])
                ep_ret_dict[f"TestEpStatsIReceived_10m_P{i + 1}"] = np.mean(received_10[i])
                ep_ret_dict[f"TestEpStatsIReceived_15m_P{i + 1}"] = np.mean(received_15[i])
                ep_ret_dict[f"TestEpStatsIReceivedPass_P{i + 1}"] = np.mean(received_p[i])
                ep_ret_dict[f"TestEpStatsIReceivedPass_5m_P{i + 1}"] = np.mean(received_p_5[i])
                ep_ret_dict[f"TestEpStatsIReceivedPass_10m_P{i + 1}"] = np.mean(received_p_10[i])
                ep_ret_dict[f"TestEpStatsIReceivedPass_15m_P{i + 1}"] = np.mean(received_p_15[i])

            logger.store(**ep_ret_dict, TestEpLen=ep_len)

        return success_rate, avg_ret

    start_time = time.time()
    o = env.reset()
    ep_ret, ep_len = np.zeros(env.num_players), 0
    total_steps = steps_per_epoch * epochs
    epoch = 0
    pkl_saved = False

    # Main loop: collect experience in env and update/log each epoch
    for t in range(total_steps):
        # Define whether to switch 1vs1 players
        switch = epoch % 2 == 0

        # Until start_steps have elapsed, randomly sample actions
        # from a uniform distribution for better exploration. Afterwards, 
        # use the learned policy (with some noise, via act_noise). 
        # Step the env
        
        if switch:
            act_1 = get_1v1_action_1(o[0][np.r_[0:18, 18:24]]) + get_1v1_action_2(o[1][np.r_[0:18, 24:30]])
        else:
            act_1 = get_1v1_action_2(o[0][np.r_[0:18, 18:24]]) + get_1v1_action_1(o[1][np.r_[0:18, 24:30]])

        if t > start_steps:
            a = act_1 + get_action(np.array(o[2:]), act_noise, (num_players - 2) // 2)
        else:
            a = act_1 + [env.action_space.sample() for _ in range(2, num_players)]

        # Step the env
        o2, r, d, _ = env.step(a)
        ep_ret += np.array(r)
        ep_len += 1

        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        d = False if ep_len==max_ep_len else d

        # Store experience to replay buffer
        if not switch:
            [replay_buffer[j].store(o[j], a[j], r[j], o2[j], d) for j in es_rb_idxs]
        else:
            [replay_buffer[2 - j - 1].store(o[j], a[j], r[j], o2[j], d) for j in es_rb_idxs if j < 2]
            [replay_buffer[j].store(o[j], a[j], r[j], o2[j], d) for j in range(2, num_players)]

        # Super critical, easy to overlook step: make sure to update 
        # most recent observation!
        o = o2

        # End of trajectory handling
        if d or (ep_len == max_ep_len):
            ep_ret_dict = {f"EpRet_P{i + 1}": ep_ret[i] for i in range(env.num_players)}
            logger.store(**ep_ret_dict, EpLen=ep_len)
            reached = False
            o, ep_ret, ep_len = env.reset(), np.zeros(env.num_players), 0

        # Update handling
        if t >= update_after and t % update_every == 0:
            for j in range(update_every):
                    
                batch_dicts = [replay_buffer[j].sample_batch(batch_size // len(es_1_idxs)) for j in es_1_idxs]
                batch = {key: np.concatenate([batch_dicts[i][key] for i in range(len(es_1_idxs))], axis=0) for key in batch_dicts[0].keys()}
                feed_dict = {x_ph: batch['obs1'],
                             x2_ph: batch['obs2'],
                             a_ph: batch['acts'],
                             r_ph: batch['rews'],
                             d_ph: batch['done']
                            }
                q_step_ops_1 = [q_loss_1, q1_1, q2_1, train_q_op_1]
                outs_q_1 = sess.run(q_step_ops_1, feed_dict)
                
                if j % policy_delay == 0:
                    # Delayed policy update
                    outs_pi_1 = sess.run([pi_loss_1, train_pi_op_1, target_update_1], feed_dict)
                    
                batch_dicts = [replay_buffer[j].sample_batch(batch_size // len(es_2_idxs)) for j in es_2_idxs]
                batch = {key: np.concatenate([batch_dicts[i][key] for i in range(len(es_2_idxs))], axis=0) for key in batch_dicts[0].keys()}
                feed_dict = {x_ph: batch['obs1'],
                             x2_ph: batch['obs2'],
                             a_ph: batch['acts'],
                             r_ph: batch['rews'],
                             d_ph: batch['done']
                            }
                q_step_ops_2 = [q_loss_2, q1_2, q2_1, train_q_op_2]
                outs_q_2 = sess.run(q_step_ops_2, feed_dict)
                logger.store(LossQ=outs_q_1[0] + outs_q_2[0], Q1Vals=outs_q_1[1] + outs_q_2[1], Q2Vals=outs_q_1[2] + outs_q_2[2])
                
                if j % policy_delay == 0:
                    # Delayed policy update
                    outs_pi_2 = sess.run([pi_loss_2, train_pi_op_2, target_update_2], feed_dict)
                    logger.store(LossPi=outs_pi_1[0] + outs_pi_2[0])


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
                    if g1 is not None: g1.finalize()
                    if g2 is not None: g2.finalize()
                success_rate = act_suc_rate
                max_ep_ret = act_avg_ret
                print("Saving model ...")
                print(f"New Best Success Rate: {int(success_rate * 100,)}, Average Episode Return: {np.round(max_ep_ret, 2)}")

            else:
                print("")

            if (((epoch % save_freq == 0) or (epoch == epochs)) and (act_suc_rate >= 0.4)) or (epoch % (save_freq * 10) == 0):
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
                    logger.log_tabular(f"TestEpStatsTeamSpreadOut_P{i + 1}")
                    logger.log_tabular(f"TestEpStatsOpIntercepted_P{i + 1}")
                    logger.log_tabular(f"TestEpStatsOpIntercepted_5m_P{i + 1}")
                    logger.log_tabular(f"TestEpStatsOpIntercepted_10m_P{i + 1}")
                    logger.log_tabular(f"TestEpStatsOpIntercepted_15m_P{i + 1}")
                    logger.log_tabular(f"TestEpStatsIReceived_P{i + 1}")
                    logger.log_tabular(f"TestEpStatsIReceived_5m_P{i + 1}")
                    logger.log_tabular(f"TestEpStatsIReceived_10m_P{i + 1}")
                    logger.log_tabular(f"TestEpStatsIReceived_15m_P{i + 1}")
                    logger.log_tabular(f"TestEpStatsIReceivedPass_P{i + 1}")
                    logger.log_tabular(f"TestEpStatsIReceivedPass_5m_P{i + 1}")
                    logger.log_tabular(f"TestEpStatsIReceivedPass_10m_P{i + 1}")
                    logger.log_tabular(f"TestEpStatsIReceivedPass_15m_P{i + 1}")

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
    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument("--gpu", type=float, default=-1)
    parser.add_argument("--reward", type=str, default="sparse")
    parser.add_argument("--control_timestep", type=float, default=0.05)
    parser.add_argument("--time_limit", type=float, default=45.)
    parser.add_argument("--use_es", type=bool, default=True)
    args = parser.parse_args()

    from spinup.utils.run_utils import setup_logger_kwargs
    es_tag = "es_" if args.use_es else ""
    logger_kwargs = setup_logger_kwargs(f'td3_soccer_2vs2_{es_tag}{args.reward}_{args.control_timestep}', data_dir=osp.join(DEFAULT_DATA_DIR, "TD3/2vs2"), datestamp=True)
    if args.gpu > 0:
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu)

        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    else:
        sess = None
    
    td3(lambda : dm_soccer2gym.make('2vs2', task_kwargs={"rew_type": args.reward, "time_limit": args.time_limit, "disable_jump": True, 
        "dist_thresh": 0.03, 'control_timestep': args.control_timestep, 'observables': 'all'}), 
        test_env_fn=lambda : dm_soccer2gym.make('2vs2', task_kwargs={"rew_type": "simple_v2", "time_limit": args.time_limit, "disable_jump": True, 
        "dist_thresh": 0.03, 'control_timestep': 0.05, 'random_state': 69, 'observables': 'all'}),
        actor_critic=core.mlp_actor_critic_heads_v2,
        gamma=args.gamma, epochs=args.epochs,
        logger_kwargs=logger_kwargs,
        sess=sess, max_ep_len=ceil(args.time_limit / args.control_timestep),
        load_1vs1=["TD3/1vs1/2020-10-08_23-06-32_td3_soccer_1vs1_dense_0.05",
                   "TD3/1vs1/2020-10-08_23-07-33_td3_soccer_1vs1_dense_0.05"],
        num=[9389999, 8629999], use_es=args.use_es)
