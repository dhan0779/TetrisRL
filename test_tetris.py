from tetris import Tetris
import random

def pretty_print(board):
    for i in range(len(board)):
        print(board[i], end='\n')
    print('\n')

env = Tetris(20, 10)
pretty_print(env.board)

ns = env.next_states()
env.next_state(random.choice(ns))
pretty_print(env.board)

env.next_state(random.choice(env.next_states()))
pretty_print(env.board)