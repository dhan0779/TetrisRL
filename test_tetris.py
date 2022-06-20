from tetris import Tetris

def pretty_print(board):
    for i in range(len(board)):
        print(board[i], end='\n')

env = Tetris(20, 10)
ns = env.next_states()

for st in ns:
    pretty_print(st[0])
    print()

print(len(ns))