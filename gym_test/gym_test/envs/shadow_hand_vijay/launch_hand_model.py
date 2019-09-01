#!/usr/bin/env python

import pybullet as p



if __name__ == '__main__':
	p.connect(p.GUI)	
	objects = p.loadSDF("./model.sdf")
	shadowHand = objects[0]
	p.resetBasePositionAndOrientation(shadowHand, [0, 0, 0],[0.7071, 0.000000, 0.000000, -0.7071])
	while(1):
		
		
		
		p.setRealTimeSimulation(1)