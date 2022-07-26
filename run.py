from tetris import Tetris
import torch
from termcolor import colored
import numpy as np
import time
import cv2
from PIL import Image

def pretty_print(board):
    colors = ["white", "red", "green", "blue", "green", "yellow", "cyan", "magenta"]
    for i in range(len(board)):
        text = ""
        for j in range(len(board[0])):
            color = colors[board[i][j]]
            text += colored(str(board[i][j]), color) + " "
        print(text)
    print('\n')

def graphics(frames):
    piece_colors = [(0, 0, 0),(240, 240, 0),(3, 240, 0),(240, 0, 1),(9, 240, 240),(160, 0, 240),(240, 160, 1),(16, 0, 240)]
    
    imgArray = []
    for i in range(len(frames)):
        img = [piece_colors[num] for row in frames[i] for num in row]
        img = np.array(img).reshape((20,10,3)).astype(np.uint8)
        img = Image.fromarray(img[..., ::-1],"RGB")
        img = np.array(img.resize((10*30, 20*30), resample=Image.BOX))
        img[[i * 30 for i in range(20)], :, :] = 0
        img[:, [i * 30 for i in range(10)], :] = 0
        imgArray.append(img)
    for img in imgArray:
        cv2.imshow('Tetris Game', img)
        cv2.setWindowProperty('Tetris Game', cv2.WND_PROP_TOPMOST, 1)
        key = cv2.waitKey(500)
        if key == 'q':#if ESC is pressed, exit loop
            cv2.destroyAllWindows()
            break

def run():
    model = torch.load("saved/model_final")
    env = Tetris(20, 10)
    outputTetris = []
    while True:
        all_states = env.next_states(env.get_next_piece())
        if len(all_states) == 0:
            break
        #pretty_print(env.board)
        #time.sleep(0.5)
        max_pred = []
        for i in range(len(all_states)):
            max_pred.append(model(torch.FloatTensor(all_states[i][1])).item())
        #print(max_pred, np.argmax(max_pred))
        ns = all_states[np.argmax(max_pred)]
        outputTetris.append(ns[0])
        env.next_state(ns)
    graphics(outputTetris)

if __name__ == "__main__":
    run()