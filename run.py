from tetris import Tetris
import torch
from termcolor import colored
import numpy as np
import time

def pretty_print(board):
    colors = ["white", "red", "green", "blue", "green", "yellow", "cyan", "magenta"]
    for i in range(len(board)):
        text = ""
        for j in range(len(board[0])):
            color = colors[board[i][j]]
            text += colored(str(board[i][j]), color) + " "
        print(text)
    print('\n')

def run():
    model = torch.load("saved/model_final")
    env = Tetris(20, 10)
    while True:
        all_states = env.next_states(env.get_next_piece())
        if len(all_states) == 0:
            break
        pretty_print(env.board)
        time.sleep(0.75)
        max_pred = []
        for i in range(len(all_states)):
            max_pred.append(model(torch.FloatTensor(all_states[i][1])).item())
        #print(max_pred, np.argmax(max_pred))
        env.next_state(all_states[np.argmax(max_pred)])

if __name__ == "__main__":
    run()