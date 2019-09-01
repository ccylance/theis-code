#!/usr/bin/env python
import random
import numpy as np
import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0,parentdir)

import gym
# from handGymEnv import HandGymEnv,Demo,unity_test
from handGymEnv import HandGymEnv,Demo
import time

#https://github.com/openai/baselines/tree/master/baselines/deepq


def main():

	hand = HandGymEnv()
	# demo = Demo()

	count = 0
	target=[2.192842715221812, 5.836050621566519e-06, 0.0588609766793831]
	action=[0,0.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
	while True:
		hand.step(action,target)
		# size=hand.getExtendedObservation()
		# print("size",len(size))
		# demo.move_finger_y_middle()

if __name__ == '__main__':
    main()


"""
Todo:
[]Generate random actions and pass it to env.step()
Deadline: 18 jun 19
"""