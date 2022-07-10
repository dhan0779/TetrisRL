from tetris import Tetris
from models.DQN2 import DQN2
import torch
import torch.nn as nn

import sys
import argparse
import numpy as np
from termcolor import colored
import time

def parse_args():
    parse_env = Tetris(20, 10)
    parser = argparse.ArgumentParser()
    parser.add_argument("--epsilon", default=parse_env.epsilon)
    parser.add_argument("--gamma", default=parse_env.gamma)
    parser.add_argument("--epochs", default=30000)
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
    optimizer = torch.optim.Adam(model.parameters())
    env = Tetris(20, 10)
    criterion = nn.MSELoss() #idk which loss tbh

    for ec in range(args.epochs):
        #time.sleep(0.25)
        #pretty_print(env.board)
        #next_states = torch.stack(zip(*env.next_states()))
        model.eval()
        with torch.no_grad():
            predicted_q = model(torch.from_numpy(np.array(env.get_metrics())))
        model.train()
        env.next_state(env.act()) #gotta fix this thing
        #print(len(env.next_states()))
        #print(predicted_q)
        #=print(torch.tensor([sum(next_state[1])]))

        optimizer.zero_grad()
        loss = criterion(predicted_q, torch.tensor([env.get_reward()]))
        loss = torch.autograd.Variable(loss, requires_grad=True)
        loss.backward()
        optimizer.step()
        
        if ec % 1000 == 0:
            torch.save(model, "saved/model_{}".format(ec))
            print("Completed Epoch {}/{} with {} games completed.".format(ec, args.epochs, env.num_games))
    
    torch.save(model, "saved/model_final")


if __name__ == "__main__":
    train(parse_args())
