from tetris import Tetris
import random
from termcolor import colored

def pretty_print(board):
    for i in range(len(board)):
        text = ""
        for j in range(len(board[0])):
            color = "white"
            if board[i][j] == 1:
                color = "red"
            elif board[i][j] == 2:
                color = "green"
            elif board[i][j] == 3:
                color = "blue"
            elif board[i][j] == 4:
                color = "green"
            elif board[i][j] == 5:
                color = "yellow"
            elif board[i][j] == 6:
                color = "cyan"
            elif board[i][j] == 7:
                color = "magenta"
            text += colored(str(board[i][j]), color) + " "
        print(text)
    print('\n')

# initialize
env = Tetris(20, 10)
pretty_print(env.board)

# # random choice
# ns = env.next_states()
# env.next_state(random.choice(ns))
# pretty_print(env.board)

for _ in range(20):
    # RL action
    env.next_state(env.act())
    pretty_print(env.board)