from tetris import Tetris
from models.DQN2 import DQN2
import torch
import torch.nn as nn

import sys
import argparse

def parse_args():
    parse_env = Tetris(20, 10)
    parser = argparse.ArgumentParser()
    parser.add_argument("--epsilon", default=parse_env.epsilon)
    parser.add_argument("--gamma", default=parse_env.gamma)
    parser.add_argument("--epochs", default=100)
    return parser.parse_args()

def train(args):
    model = DQN2()
    optimizer = torch.optim.Adam(model.parameters())
    env = Tetris(20, 10)
    
    for _ in range(args.epochs):
        next_state = env.act()
        

if __name__ == "__main__":
    train(parse_args())
