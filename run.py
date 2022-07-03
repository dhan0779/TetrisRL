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

    while env.max_height < env.height:
        pretty_print(env.board)
        time.sleep(0.5)
        max_pred = []
        all_states = env.next_states()
        for  i in range(len(all_states)):
            max_pred.append(model(torch.from_numpy(np.array(all_states[i][1]))).item())
        env.next_state(all_states[np.argmax(max_pred)])

if __name__ == "__main__":
    run()