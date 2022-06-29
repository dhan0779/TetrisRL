import numpy as np
import random

class Tetris:
    pieces = [
        [[1,1],
         [1,1]],
        [[0,2,2],
         [2,2,0]],
        [[3,3,0],
         [0,3,3]],
        [[4,4,4,4]],
        [[0,5,0],
         [5,5,5]],
        [[0,0,6],
         [6,6,6]],
        [[7,0,0],
         [7,7,7]]
    ]

    def __init__(self, height, width):
        self.height = height
        self.width = width
        self.reset_state()

    def reset_state(self):
        self.board = [[0] * self.width for _ in range(self.height)]
        self.piece = -1 #use indicies to keep track of piece in pieces
        self.lines_cleared = 0
        self.max_height = 0
        self.bumpy = 0

    def get_next_piece(self):
        self.piece = random.randint(0, len(self.pieces)-1)
        return self.piece
    
    def rotate_piece(self, piece):
        piece = np.array(piece)
        piece = np.transpose(piece)
        return piece[..., ::-1]

    def next_states(self):
        all_states = [] #key: board, value: properties changed
        ind_piece = self.get_next_piece()
        next_piece = self.pieces[ind_piece]
        
        rotations = 1
        if ind_piece in [1,2,3]: 
            rotations = 2
        if ind_piece in [4,5,6]:
            rotations = 4
        
        for _ in range(rotations):
            for i in range(self.width-len(next_piece[0])+1):
                j = 0
                while self.check_valid_move((j,i), next_piece):
                    j+=1
                nboard, rows_cleared, max_height, bumpy = self.put_piece((j-1, i), next_piece)
                all_states.append((nboard, [rows_cleared, max_height, bumpy]))
            next_piece = self.rotate_piece(next_piece)
        return all_states

    def check_valid_move(self, pos, piece): #check if putting piece is valid
        for i in range(len(piece)):
            for j in range(len(piece[0])):
                if pos[0]+i >= self.height or pos[1]+j >= self.width: 
                    return False
                if piece[i][j] != 0 and self.board[pos[0]+i][pos[1]+j] != 0:
                    return False
        return True
    
    def put_piece(self, pos, piece): #actually put the damn piece
        board = [i[:] for i in self.board]
        for i in range(len(piece)):
            for j in range(len(piece[0])):
                board[pos[0]+i][pos[1]+j] = piece[i][j]
        
        num_cleared = 0
        n_board = []
        for row in board:
            if 0 not in row:
                num_cleared+=1
            else: n_board.append(row)
        for _ in range(num_cleared):
            board.insert(0, [0]*self.width)

        max_height = max(self.max_height, pos[1])

        bumpy = 0
        prev_height = -1
        r = 0
        c = 0
        while c < len(board[0]):
            if r == len(board) - 1:
                if prev_height == -1:
                    prev_height = r
                bumpy += (prev_height - r)
                prev_height = r
                r = 0
                c += 1
            elif board[r][c] == 0:
                r += 1
            else:
                if prev_height == -1:
                    prev_height = r
                bumpy += (prev_height - r)
                prev_height = r
                r = 0
                c += 1
            
        return board, num_cleared, max_height, bumpy
    
    def next_state(self, state): #might change based on other features
        self.board = state[0]

        self.lines_cleared += state[1][0]
        self.max_height = state[1][1]
        self.bumpy = state[1][2]
