
import gym
import linet
import train

import argparse
import torch
import random

LAYER_SIZE = 16 # number of nodes per layer
NUM_HIDDEN = 4 # number of hidden layers

EPSILON_SCHEDULE = 1000
MAX_TIME = 2000
GAMMA = .99

STEPS_PER = 100
SAMPLE_FREQ = 10
FIRST_TRIALS = 100

def main(args):

    # useful for debugging
    if args.full_print:
        torch.set_printoptions(profile="full")

    # get the correct device
    device = torch.device("cpu")
    if args.cuda:
        device = torch.device('cuda:0')

    env = gym.make("MountainCar-v0")

    net = linet.LiNet(LAYER_SIZE, NUM_HIDDEN, device=device, last_activates=False, act_func='elu')

    epsilon = 1

    input_buffer = []
    action_buffer = []
    reward_buffer = []
    first_trial = True
    iteration = 0
    while True:
        iteration += 1
        print(" --- Iteration", iteration, "---")

        for trial in range(FIRST_TRIALS if first_trial else 1):
            first_trial = False

            obs = env.reset()[0]
            net.reset()

            inputs = []
            actions = []
            rewards = []

            action = None

            done = False
            t = -1
            while not done and t <= MAX_TIME:
                t += 1

                pos = torch.from_numpy(obs)[0]

                if t % SAMPLE_FREQ == 0: inputs.append(pos)

                in_tensor = torch.zeros(LAYER_SIZE, device=device)
                in_tensor[0] = pos

                out = net.forward(in_tensor)[:3]

                if t % SAMPLE_FREQ == 0:
                    action = random.randrange(3)
                    if random.random() > epsilon:
                        action = torch.argmax(out)
                    actions.append(action)

                obs, reward, done, something, info = env.step(int(action))
                if t % SAMPLE_FREQ == 0:
                    rewards.append(reward)
                
        
            if sum(rewards) < -100:
                continue

            print("Total Reward:", sum(rewards))

            x = torch.zeros((len(inputs), LAYER_SIZE), device=device)
            for t in range(len(inputs)):
                x[t][0] = inputs[t]
            input_buffer.append(x)

            action_buffer.append(actions)
            reward_buffer.append(rewards)

        if epsilon > 0:
            epsilon -= 1/EPSILON_SCHEDULE

        x_data = []
        y_data = []
        for i in range(len(input_buffer)):
            x_data.append(input_buffer[i])
            x = x_data[-1]

            preds = []
            targets = []

            out = None
            net.reset()
            for t in range(x.shape[0]):
                out = net.forward(x[t])[:3]

                preds.append(out[action_buffer[i][t]])
                if t > 0:
                    targets[-1][1] += GAMMA * torch.max(out)
                targets.append([action_buffer[i][t], reward_buffer[i][t]])
            
            y_data.append(targets)

        print("Avg Target:", sum([sum([target[i][1] for i in range(len(target))]) / len(target) for target in y_data]) / len(y_data))

        total_loss = 0
        num_points = 0
        for step in range(STEPS_PER):

            i = random.randrange(len(x_data))
            x = x_data[i]
            y = y_data[i]

            grads = torch.zeros_like(x)

            out = None
            net.reset()
            for t in range(len(y)):
                out = net.forward(x[t])[:3]

                pred = out[y[t][0]]

                total_loss += (y[t][1] - pred)**2
                num_points += 1
                grads[t][action_buffer[i][t]] = (y[t][1] - pred)

            net.backward(x, grads, max_tau=min(round(.75*len(y)), linet.DEFAULT_MAX_TAU))

            net.apply_grads(1e-8)

        print("Avg Loss:", (total_loss/num_points).item())

        if iteration % 100 == 0:
            net.save(train.CHECKPOINT_FOLDER+"iteration_"+str(iteration)+"_gammas.pt", train.CHECKPOINT_FOLDER+"iteration_"+str(iteration)+"_gains.pt")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Python-based choicenet')

    parser.add_argument('--gpu', dest='cuda', action='store_const', const=True, default=False, 
                    help='Whether to use cuda gpu acceleration')
    parser.add_argument('--full', dest='full_print', action='store_const', const=True, default=False, 
                    help='Whether to print entire tensors')

    args = parser.parse_args()
    main(args)