import numpy as np
import random


class Tetris:
    pieces = [
        [[1, 1],
         [1, 1]],
        [[0, 2, 2],
         [2, 2, 0]],
        [[3, 3, 0],
         [0, 3, 3]],
        [[4, 4, 4, 4]],
        [[0, 5, 0],
         [5, 5, 5]],
        [[0, 0, 6],
         [6, 6, 6]],
        [[7, 0, 0],
         [7, 7, 7]]
    ]

    def __init__(self, height, width):
        self.height = height
        self.width = width
        self.epsilon = 1
        self.epsilon_floor = 0.1  # minimum possible value for epsilon
        self.gamma = 0.99999975  # epsilon decay rate
        self.num_games = 0
        self.reset_state()

    def reset_state(self):
        self.board = [[0] * self.width for _ in range(self.height)]
        self.piece = -1  # use indicies to keep track of piece in pieces
        self.lines_cleared = 0
        self.max_height = 0
        self.bumpy = 0
        self.holes = 0
        self.score = 0
        self.turns = 0
        self.states = []

    def get_next_piece(self):
        self.piece = random.randint(0, len(self.pieces)-1)
        return self.piece

    def rotate_piece(self, piece):
        piece = np.array(piece)
        piece = np.transpose(piece)
        return piece[..., ::-1]

    def count_holes(self, orig_board):
        board = [i[:] for i in orig_board]

        def fill(x, y, start_color, new_color):
            if board[x][y] != start_color:
                return
            elif board[x][y] == new_color:
                return
            else:
                board[x][y] = new_color
                neighbors = [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]
                for n in neighbors:
                    if 0 <= n[0] <= len(board)-1 and 0 <= n[1] <= len(board[0])-1:
                        fill(n[0], n[1], start_color, new_color)
        for j in range(len(board[0])):
            fill(0, j, 0, 9)

        holes = 0
        for i in range(len(board)):
            for j in range(len(board[0])):
                if board[i][j] == 0:
                    holes += 1

        return holes

    def next_states(self):
        all_states = []  # key: board, value: properties changed
        ind_piece = next_piece
        next_piece = self.pieces[ind_piece]

        rotations = 1
        if ind_piece in [1, 2, 3]:
            rotations = 2
        if ind_piece in [4, 5, 6]:
            rotations = 4

        for _ in range(rotations):
            for i in range(self.width-len(next_piece[0])+1):
                j = 0
                while self.check_valid_move((j, i), next_piece):
                    j += 1
                nboard, rows_cleared, max_height, bumpy, holes, score = self.put_piece(
                    (j-1, i), next_piece)
                all_states.append(
                    (nboard, [rows_cleared, max_height, bumpy, holes, score]))
            next_piece = self.rotate_piece(next_piece)
        self.states = all_states
        return all_states

    def check_valid_move(self, pos, piece):  # check if putting piece is valid
        for i in range(len(piece)):
            for j in range(len(piece[0])):
                if pos[0]+i >= self.height or pos[1]+j >= self.width:
                    return False
                if piece[i][j] != 0 and self.board[pos[0]+i][pos[1]+j] != 0:
                    return False
        return True

    def put_piece(self, pos, piece):  # actually put the damn piece
        # scoring
        this_score = 8

        board = [i[:] for i in self.board]
        for i in range(len(piece)):
            for j in range(len(piece[0])):
                if board[pos[0]+i][pos[1]+j] == 0:
                    board[pos[0]+i][pos[1]+j] = piece[i][j]

        num_cleared = 0
        n_board = []
        for row in board:
            if 0 not in row:
                num_cleared += 1
            else:
                n_board.append(row)
        for _ in range(num_cleared):
            n_board.insert(0, [0]*self.width)

        # scoring for lines cleared
        if num_cleared > 0:
            multipliers = [40, 100, 300, 1200]
            this_score += int(multipliers[num_cleared - 1]
                              * ((self.lines_cleared / 10) + 1))

        max_height = max(self.max_height, self.height - pos[0])

        bumpy = 0
        prev_height = -1
        r = 0
        c = 0
        while c < len(n_board[0]):
            if r == len(n_board) - 1:
                if prev_height == -1:
                    prev_height = r
                bumpy += abs(prev_height - r)
                prev_height = r
                r = 0
                c += 1
            elif n_board[r][c] == 0:
                r += 1
            else:
                if prev_height == -1:
                    prev_height = r
                bumpy += abs(prev_height - r)
                prev_height = r
                r = 0
                c += 1

        num_holes = self.count_holes(n_board)

        return n_board, num_cleared, max_height, bumpy, num_holes, self.score + this_score

    def next_state(self, state):  # might change based on other features
        self.board = state[0]
        self.lines_cleared += state[1][0]
        self.max_height = state[1][1]
        self.bumpy = state[1][2]
        self.holes = state[1][3]
        self.score = state[1][4]

    def get_metrics(self):
        return [self.lines_cleared, self.score, self.max_height, self.bumpy, self.holes]

    def get_reward(self):
        return 20 * self.lines_cleared + self.score - self.max_height - self.bumpy - self.holes

    def act(self):
        self.turns += 1
        # Given a state, choose an epsilon-greedy action
        states = self.next_states(self.get_next_piece())
        if len(states) == 0:
            self.reset_state()
            self.num_games+=1
            states = self.next_states(self.get_next_piece())
        # EXPLORE: Rather than use the learning model, just randomly choose a next state
        if np.random.rand() < self.epsilon:
            idx = np.random.randint(0, len(states) - 1)
        # EXPLOITATION: Use the greedy strategy: choose the state with the max number of rows cleared. Side note: I implemented way to check number of holes but idk how we want to factor that in yet (discuss with Emmett)
        else:
            reward_per_state = []
            idx = -1
            for i in range(len(states)):
                # for readability
                lines_cleared = states[i][1][0]
                max_height = states[i][1][1]
                bumps = states[i][1][2]
                holes = states[i][1][3]
                score = states[i][1][4]
                # reward function
                reward_per_state.append(
                    20 * lines_cleared + score - max_height - bumps - holes)
            idx = np.argmax(reward_per_state)

        self.epsilon *= self.gamma
        # Want to make sure epsilon never falls below a certain rate
        self.gamma = max(self.epsilon_floor, self.epsilon)

        return states[idx]

    def play_game(self):
        play = True
        while play:
            self.next_state(self.act())
            if self.max_height >= self.height:
                play = False
