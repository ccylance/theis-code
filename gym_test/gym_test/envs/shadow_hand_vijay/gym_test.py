#!/usr/bin/env python

import pybullet as p
import random
import numpy as np
from mamad_util import JointInfo

def check_collision(active_joints_info,num_active_joints):
	collision_set=[]
	index_of_active_joints = [active_joints_info[i]["jointIndex"] for i in range(num_active_joints)]
	for i in index_of_active_joints:
		for j in index_of_active_joints:
			if i == j:
				continue
			contact = p.getClosestPoints(fingerID,fingerID,0,i,j)
			
			if len(contact)!=0:
				collision_set.append([contact[0][3],contact[0][4]])
	
	check_flip=[]
	for i in range(len(collision_set)):
		index_1=collision_set[i][0]
		index_2=collision_set[i][1]
		for j in range(i,len(collision_set)):
			if i == j:
				continue
			if index_1 == collision_set[j][1] and index_2 ==  collision_set[j][0]:
				check_flip.append(j)
	
	new_check=[]
	sort=np.argsort(check_flip)
	for i in range(len(check_flip)):
		new_check.append(check_flip[sort[i]])
	for i in range(len(check_flip)):
		del collision_set[new_check[i]-i]
	
	check_parent=[]
	for i in range(len(parent_list)):
		index_parent_1=parent_list[i][0]
		index_parent_2=parent_list[i][1]
		for j in range(len(collision_set)):
			if index_parent_1 == collision_set[j][0] and index_parent_2 ==  collision_set[j][1]:
				check_parent.append(j)
			if index_parent_1 == collision_set[j][1] and index_parent_2 ==  collision_set[j][0]:
				check_parent.append(j)
	new_check_parent=[]
	sort_parent=np.argsort(check_parent)
	for i in range(len(check_parent)):
		new_check_parent.append(check_parent[sort_parent[i]])
	for i in range(len(check_parent)):
		del collision_set[new_check_parent[i]-i]

	collision_result=[]
	for i in range (len(collision_set)):
		index_collision_set_1=collision_set[i][0]
		index_collision_set_2=collision_set[i][1]
		for j in range(num_active_joints):
			if index_collision_set_1 == active_joints_info[j]["jointIndex"]:
				index_collision_set_1_result = j
			if index_collision_set_2 == active_joints_info[j]["jointIndex"]:
				index_collision_set_2_result = j	

		collision_result.append([active_joints_info[index_collision_set_1_result]["linkName"],active_joints_info[index_collision_set_2_result]["linkName"]])
	return collision_result

p.connect(p.GUI)
p.setGravity(0,0,-9.8)
finger = p.loadSDF("./model.sdf")
fingerID = finger[0]

jointInfo = JointInfo()
jointInfo.get_infoForAll_joints(finger)
active_joints_info  = jointInfo.getActiveJointsInfo()
num_active_joints = jointInfo.getNumberOfActiveJoints()
num_joints = p.getNumJoints(fingerID)

 

# print("active_joints_info::",active_joints_info)
# print("finger::",finger)

# print("`num of joints:::",num_joints)
"""
for i in range(num_joints):
	j_info = p.getJointInfo(fingerID,i)
	print("joint_info::",j_info)
"""
# texUid = p.loadTexture("./../cube_new/aaa.png")
# cube_objects = p.loadSDF("./../cube_new/model.sdf")
# p.changeVisualShape(cube_objects[0], -1, rgbaColor=[1, 1, 1, 1])
# p.changeVisualShape(cube_objects[0], -1, textureUniqueId=texUid)
# p.resetBasePositionAndOrientation(cube_objects[0], [0, 0.37, 0.07],[0.7071, 0.000000, 0.000000, 0.7071])
p.setRealTimeSimulation(0)
p.setTimeStep(1./5000)
while(1):
	p.resetBasePositionAndOrientation(fingerID, [0, 0, 0],[0.7071, 0.000000, 0.000000, -0.7071])
	parent_list=[]
	for i in range(num_active_joints):
		jointIndex = active_joints_info[i]["jointIndex"]
		jointName = active_joints_info[i]["jointName"]
		linkName = active_joints_info[i]["linkName"]
		jointPositionState = p.getJointState(fingerID,jointIndex)[0]
		# print("linkName::",linkName)
		# print("jointName::",jointName)
		# print("jointIndex::",jointIndex)
		# print("jointPositionState::",jointPositionState)
		jointll = active_joints_info[i]["jointLowerLimit"]
		jointul = active_joints_info[i]["jointUpperLimit"]
		# print("lower limit",jointll)
		# print("upper limit",jointul)
		motor_command = jointPositionState
		parent_list.append([jointIndex,jointInfo.searchBy("jointIndex",jointIndex)[0]["parentIndex"]])
		if jointIndex == 3:
		
			step =(abs(jointll)-abs(jointul))/100
			motor_command = jointPositionState+0.0
		
		p.setJointMotorControl2(fingerID,jointIndex,p.POSITION_CONTROL,motor_command, force=1.0)

	collision_result=check_collision(active_joints_info,num_active_joints)

	#print("right hand self coliision -------",collision_set)
	print("right hand self coliision -------",collision_result)
	print("\n")
	p.stepSimulation()




