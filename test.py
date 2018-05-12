import numpy as np
import sys
sys.path.append("game/")
import wrapped_flappy_bird as game
import cv2
import random


def pre_process(img):
    x_t = cv2.cvtColor(cv2.resize(img, (80, 80)), cv2.COLOR_BGR2GRAY)
    ret, x_t = cv2.threshold(x_t, 1, 255, cv2.THRESH_BINARY)

    return x_t




game_state = game.GameState()
total_steps = 0
temp_action = random.randint(0, 1)
action = np.zeros([2])
action[temp_action] = 1
new_state, reward, done = game_state.frame_step(action)

total_steps += 1

temp_img = pre_process(new_state)
img_batch = [temp_img] * 4



print (temp_img)