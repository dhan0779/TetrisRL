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
                color = "grey"
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

# initialize
env = Tetris(20, 10)
env.play_game()
pretty_print(env.board)
print("score: " + str(env.score))
print("turns: " + str(env.turns))
print("lines cleared: " + str(env.lines_cleared))
print("max height: " + str(env.max_height))
print("bumpyness: " + str(env.bumpy))
print("holes: " + str(env.holes))
print('\n')