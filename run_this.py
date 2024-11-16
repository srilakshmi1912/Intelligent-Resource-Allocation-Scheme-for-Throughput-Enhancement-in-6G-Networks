from env import Env
from DDPG import DDPG
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from datetime import datetime

#####################  hyper parameters  ####################
CHECK_EPISODE = 4   
LEARNING_MAX_EPISODE = 100
MAX_EP_STEPS = 1000
TEXT_RENDER = True
SCREEN_RENDER = False
CHANGE = False
SLEEP_TIME = 0.1
 
#####################  function  ####################
def exploration(a, r_dim, b_dim, r_var, b_var):
    for i in range(r_dim + b_dim):
        # resource
        if i < r_dim:
            a[i] = np.clip(np.random.normal(a[i], r_var), 0, 1) * r_bound
        # bandwidth
        elif i < r_dim + b_dim:
            a[i] = np.clip(np.random.normal(a[i], b_var), 0, 1) * b_bound
    return a

###############################  training  ####################################

if __name__ == "__main__":
    env = Env()

    s_dim, r_dim, b_dim, o_dim, r_bound, b_bound, task_inf, limit, location = env.get_inf()
    ddpg = DDPG(s_dim, r_dim, b_dim, o_dim, r_bound, b_bound)

    r_var = 1  # control exploration
    b_var = 1
    r_v, b_v = [], []

    ep_reward = []
    ep_throughput = []
    ep_edge_capabilities = []  # Store edge capabilities for each episode
    ep_offload = []
    ep_migration_size = []

    var_reward = []
    max_rewards = 0
    episode = 0
    var_counter = 0
    epoch_inf = []

    episode_rewards = []
    episode_times = []

    #####################################################
    now = datetime.now()
    current_time = now.strftime("%H_%M_%S")
    # make directory
    dir_name = 'output_episodes/' + 'ddpg_'+str(r_dim) + 'u' + str(int(o_dim / r_dim)) + 'e' + str(limit) + 'l' + location+'t'+current_time
    if (os.path.isdir(dir_name)):
        os.rmdir(dir_name)
    os.makedirs(dir_name)
    #####################################################
    while var_counter < LEARNING_MAX_EPISODE:
        # initialize
        s = env.reset()
        ep_reward.append(0)
        ep_throughput.append(0)
        ep_edge_capabilities.append(0)
        ep_offload.append(0)
        ep_migration_size.append(0)

        if SCREEN_RENDER:
            env.initial_screen_demo()

        for j in range(MAX_EP_STEPS):
            start_time = time.time()  
            time.sleep(SLEEP_TIME)
            # render 
            if SCREEN_RENDER:
                env.screen_demo()
            if TEXT_RENDER and j % 30 == 0:
                env.text_render()

            # DDPG
            # choose action according to state
            a = ddpg.choose_action(s)  # a = [R B O]
            # add randomness to action selection for exploration
            a = exploration(a, r_dim, b_dim, r_var, b_var)
            # store the transition parameter
            s_, r, edge_capability_sum, throughput, offload, migration_size = env.ddpg_step_forward(a, r_dim, b_dim)  
             

            print("EPISODE:", episode)
            print(edge_capability_sum, throughput)

            ddpg.store_transition(s, a, r / 10, s_)

            # learn
            if ddpg.pointer == ddpg.memory_capacity:
                print("start learning")
            if ddpg.pointer > ddpg.memory_capacity:
                ddpg.learn()
                if CHANGE:
                    r_var *= .99999
                    b_var *= .99999
            # replace the state
            s = s_
            # sum up the reward
            ep_reward[episode] += r
            ep_edge_capabilities[episode] += edge_capability_sum
            ep_throughput[episode] += throughput 
            ep_offload[episode] += offload
            ep_migration_size[episode] += migration_size
            
            # in the end of the episode
            if j == MAX_EP_STEPS - 1:
                var_reward.append(ep_reward[episode])
                #################
                filename = '/episode_'+str(episode)+'.txt'
                f1 = open(dir_name + filename, 'a')
                f1.write("reward :"+str(ep_reward[episode]) + "\n")
                f1.write("throughput :"+str(ep_throughput[episode]) + "\n")
                f1.write("edge capability :"+str(ep_edge_capabilities[episode]) + "\n")
                f1.write("no of edges offloading :"+str(ep_offload[episode]) + "\n")
                f1.write("migration size :"+str(ep_migration_size[episode]) + "\n")
                
                f1.close()
                ##################


                r_v.append(r_var)
                b_v.append(b_var)
                print('Episode:%3d' % episode, ' Reward: %5d' % ep_reward[episode], '###  r_var: %.2f ' % r_var, 'b_var: %.2f ' % b_var, ' Throughput: %5d' % ep_throughput[episode], ' Edge capability: %5d' % ep_edge_capabilities[episode], ' Offload_no: %5d' % ep_offload[episode], ' Migration_size: %5d' % ep_migration_size[episode])
                string = 'Episode:%3d' % episode + ' Reward: %5d' % ep_reward[episode] + '###  r_var: %.2f ' % r_var + 'b_var: %.2f ' % b_var + ' Throughput: %5d' % ep_throughput[episode] + ' Edge capability: %5d' % ep_edge_capabilities[episode] + ' Offload_no: %5d' % ep_offload[episode] +' Migration_size: %5d' % ep_migration_size[episode]
                epoch_inf.append(string)
                # variation change
                if var_counter >= CHECK_EPISODE and np.mean(var_reward[-CHECK_EPISODE:]) >= max_rewards:
                    CHANGE = True
                    var_counter = 0
                    max_rewards = np.mean(var_reward[-CHECK_EPISODE:])
                    var_reward = []
                else:
                    CHANGE = False
                    var_counter += 1

        # end the episode
        if SCREEN_RENDER:
            env.canvas.tk.destroy()
        episode += 1

    # make directory
    dir_name = 'output/' + 'ddpg_'+str(r_dim) + 'u' + str(int(o_dim / r_dim)) + 'e' + str(limit) + 'l' + location+'t'+current_time
    if (os.path.isdir(dir_name)):
        os.rmdir(dir_name)
    os.makedirs(dir_name)

    # plot the reward
    fig_reward = plt.figure()
    plt.plot([i+1 for i in range(episode)], ep_reward)
    plt.xlabel("episode")
    plt.ylabel("rewards")
    fig_reward.savefig(dir_name + '/rewards.png')

    # plot the throughput
    fig_reward = plt.figure()
    plt.plot([i+1 for i in range(episode)], ep_throughput)
    plt.xlabel("episode")
    plt.ylabel("throughput")
    fig_reward.savefig(dir_name + '/throughput.png')

    # plot the variance
    fig_variance = plt.figure()
    plt.plot([i + 1 for i in range(episode)], r_v, b_v)
    plt.xlabel("episode")
    plt.ylabel("variance")
    fig_variance.savefig(dir_name + '/variance.png')

    # Plot episode vs edge capabilities
    fig_edge_capabilities = plt.figure()
    plt.plot([i + 1 for i in range(episode)], ep_edge_capabilities)
    plt.xlabel("episode")
    plt.ylabel("edge capabilities")
    fig_edge_capabilities.savefig(dir_name + '/edge_capabilities.png')

    # write the record
    f = open(dir_name + '/record.txt', 'a')
    f.write('time(s):' + str(MAX_EP_STEPS) + '\n\n')
    f.write('user_number:' + str(r_dim) + '\n\n')
    f.write('edge_number:' + str(int(o_dim / r_dim)) + '\n\n')
    f.write('limit:' + str(limit) + '\n\n')
    f.write('task information:' + '\n')
    f.write(task_inf + '\n\n')
    for i in range(episode):
        f.write(epoch_inf[i] + '\n')

    # mean
    print("the mean of the rewards in the last", LEARNING_MAX_EPISODE, " epochs:", str(np.mean(ep_reward[-LEARNING_MAX_EPISODE:])))
    f.write("the mean of the rewards:" + str(np.mean(ep_reward[-LEARNING_MAX_EPISODE:])) + '\n\n')
    # standard deviation
    print("the standard deviation of the rewards:", str(np.std(ep_reward[-LEARNING_MAX_EPISODE:])))
    f.write("the standard deviation of the rewards:" + str(np.std(ep_reward[-LEARNING_MAX_EPISODE:])) + '\n\n')
    # range
    print("the range of the rewards:", str(max(ep_reward[-LEARNING_MAX_EPISODE:]) - min(ep_reward[-LEARNING_MAX_EPISODE:])))
    f.write("the range of the rewards:" + str(max(ep_reward[-LEARNING_MAX_EPISODE:]) - min(ep_reward[-LEARNING_MAX_EPISODE:])) + '\n\n')
    f.close()
