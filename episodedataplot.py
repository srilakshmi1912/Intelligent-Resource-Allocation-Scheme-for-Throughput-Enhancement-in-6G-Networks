import matplotlib.pyplot as plt
import os
from datetime import datetime

now = datetime.now()
nameoff = "ddpg_10u20e4lKAISTt23_26_04"
current_time = now.strftime("%H_%M_%S")
dir_name = 'output_episodes/' + nameoff
ep_reward = {} 
ep_throughput = {}
ep_edge_capabilities = {}
ep_offload = {}
ep_migration_size = {}

n_episodes = 0

for filename in os.listdir(dir_name):
    f = os.path.join(dir_name, filename)

    # Check if it is a file and has the expected format
    if os.path.isfile(f) and filename.startswith("episode") and filename.endswith(".txt"):
        episode = int(f.split(".")[0].split("_")[5])

        n_episodes += 1
        with open(f, 'r') as fi:
            content = fi.read()

        reward = float(content.split("reward :")[1].split("throughput")[0])
        throughput = float(content.split("throughput :")[1].split("edge capability")[0])
        edge_capability_sum = float(content.split("edge capability :")[1].split("no of edges offloading")[0])
        offload = float(content.split("no of edges offloading :")[1].split("migration size")[0])
        migration_size = float(content.split("migration size :")[1])


        ep_reward[episode] = reward
        ep_edge_capabilities[episode] = edge_capability_sum
        ep_throughput[episode] = throughput
        ep_offload[episode] = offload
        ep_migration_size[episode] = migration_size


        # Print the information for the current episode
        # print('Episode:%3d' % episode, ' Reward: %5d' % ep_reward[episode], ' Throughput: %5d' % ep_throughput[episode], ' Edge capability: %5d' % ep_edge_capabilities[episode])
        # Store the information in epoch_inf
        string = 'Episode:%3d' % episode + ' Reward: %5d' % ep_reward[episode] + ' Throughput: %5d' % ep_throughput[episode] + ' Edge capability: %5d' % ep_edge_capabilities[episode] + ' Offload_no: %5d' % ep_offload[episode] +' Migration_size: %5d' % ep_migration_size[episode]

        # Write the record for each episode
        with open(dir_name + '/record.txt', 'a') as f:
            f.write(string + '\n')

print('plots and record.txt has been created inside ' + nameoff)

# Plot episode vs rewards
fig_reward = plt.figure()
plt.plot([i + 1 for i in range(n_episodes)], [ep_reward[i] for i in range(n_episodes)])
plt.xlabel("episode")
plt.ylabel("rewards")
fig_reward.savefig(dir_name + '/plot_rewards.png')

# Plot episode vs throughput
fig_throughput = plt.figure()
plt.plot([i + 1 for i in range(n_episodes)], [ep_throughput[i] for i in range(n_episodes)])
plt.xlabel("episode")
plt.ylabel("throughput")
fig_throughput.savefig(dir_name + '/plot_throughput.png')

# Plot episode vs edge capabilities
fig_edge_capabilities = plt.figure()
plt.plot([i + 1 for i in range(n_episodes)], [ep_edge_capabilities[i] for i in range(n_episodes)])
plt.xlabel("episode")
plt.ylabel("edge capabilities")
fig_edge_capabilities.savefig(dir_name + '/plot_edge_capabilities.png')
