from tetris import Tetris
from models.DQN2 import DQN2
import torch
import torch.nn as nn

import sys
import argparse
import numpy as np
from termcolor import colored
import random

def parse_args():
    parse_env = Tetris(20, 10)
    parser = argparse.ArgumentParser()
    parser.add_argument("--epsilon", default=parse_env.epsilon)
    parser.add_argument("--gamma", default=parse_env.gamma)
    parser.add_argument("--epochs", default=30000)
    parser.add_argument("--replay_size", default=100000)
    parser.add_argument("--minibatch_size", default=200)
    return parser.parse_args()

def pretty_print(board):
    colors = ["white", "red", "green", "blue", "green", "yellow", "cyan", "magenta"]
    for i in range(len(board)):
        text = ""
        for j in range(len(board[0])):
            color = colors[board[i][j]]
            text += colored(str(board[i][j]), color) + " "
        print(text)
    print('\n')

def train(args):
    model = DQN2()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    env = Tetris(20, 10)
    criterion = nn.MSELoss() #idk which loss tbh
    replay = []

    for ec in range(args.epochs):
        #time.sleep(0.25)
        #pretty_print(env.board)
        #next_states = torch.stack(zip(*env.next_states()))
        model.eval()
        state = torch.FloatTensor(env.get_metrics())
        all_states = env.next_states(env.get_next_piece())
        terminal = False

        with torch.no_grad():
            output = model(torch.FloatTensor(list(zip(*all_states))[1]))
        model.train()
        idx = env.act(all_states, output)
        
        env.next_state(all_states[idx]) 
        if env.max_height == env.height or env.max_height+1 == env.height:
            env.reset_state()
            env.num_games+=1
            terminal = True

        replay.append((state, env.get_reward(), torch.FloatTensor(env.get_metrics())))
        if len(replay) > args.replay_size:
            replay.pop(0)

        minibatch = random.sample(replay, min(len(replay), args.minibatch_size))
        state_batch = torch.stack(tuple(d[0] for d in minibatch))
        reward_batch = torch.from_numpy(np.array([d[1] for d in minibatch]))
        state_after_batch = torch.stack(tuple(d[2] for d in minibatch))

        q_values = model(state_batch)
        model.eval()
        with torch.no_grad():
            next_pred_batch = model(state_after_batch) 
        model.train()
        #print(reward_batch[0]+ env.gamma* torch.max(next_pred_batch[0]))
        y_batch = torch.stack(tuple(reward_batch[i] if terminal else reward_batch[i]+env.gamma*torch.max(next_pred_batch[i]) for i in range(len(minibatch))))
        loss = criterion(q_values, y_batch)
        loss = torch.autograd.Variable(loss, requires_grad=True)
        print("Loss: {}, Reward: {}, Epsilon: {}".format(loss.item(), env.get_reward(), env.epsilon))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if ec % 1000 == 0:
            torch.save(model, "saved/model_{}".format(ec))
            print("Completed Epoch {}/{} with {} games completed and epsilon is {}.".format(ec, args.epochs, env.num_games, env.epsilon))
    
    torch.save(model, "saved/model_final")


if __name__ == "__main__":
    train(parse_args())
