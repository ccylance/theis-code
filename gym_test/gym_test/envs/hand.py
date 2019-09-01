import os,  inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0,parentdir)

import pybullet as p
import numpy as np
import copy
import math
import pybullet_data
import time
from pkg_resources import resource_string,resource_filename

from mamad_util import JointInfo


class Hand:

  def __init__(self, urdfRootPath=pybullet_data.getDataPath(), timeStep=0.01):
    self.urdfRootPath = urdfRootPath
    self.timeStep = timeStep
    self.maxVelocity = .35
    self.maxForce = 200.
    self.fingerAForce = 2 
    self.fingerBForce = 2.5
    self.fingerTipForce = 2
    self.useSimulation = 1
    self.useNullSpace =21
    self.useOrientation = 1
    self.handId =3
    self.kukaEndEffectorIndex = 6
    self.kukaGripperIndex = 7
    self.jointInfo = JointInfo()
    self.fingerTip_joint_name = ["J1_FF","J1_MF","J1_RF","J1_LF","THJ1"] #this are joints between final link and one before it
    self.wrist_joint_name = ["WRJ2","WRJ1"]
    #lower limits for null space
    self.ll=[-.967,-2 ,-2.96,0.19,-2.96,-2.09,-3.05]
    #upper limits for null space
    self.ul=[.967,2 ,2.96,2.29,2.96,2.09,3.05]
    #joint ranges for null space
    self.jr=[5.8,4,5.8,4,5.8,4,6]
    #restposes for null space
    self.rp=[0,0,0,0.5*math.pi,0,-math.pi*0.5*0.66,0]
    #joint damping coefficents
    self.jd=[0.00001,0.00001,0.00001,0.00001,0.00001,0.00001,0.00001,0.00001,0.00001,0.00001,0.00001,0.00001,0.00001,0.00001]
    self.reset()


  def get_fingerTips_linkIndex(self):
    fingerTips_linkIndex = []
    fingerTips_jointInfo = []
    fingerTip_joint_name_bytes = [i.encode(encoding='UTF-8',errors='strict') for i in self.fingerTip_joint_name]
    #print("Debug11::fingerTip_joint_name_bytes",fingerTip_joint_name_bytes)
    # getting joints for the final link
    # print(fingerTip_joint_name_bytes)
    for i in fingerTip_joint_name_bytes:
      # print(i,"   ",self.jointInfo.searchBy(key="jointName",value = i))
      fingerTips_jointInfo.append(self.jointInfo.searchBy(key="jointName",value = i))
      
    #extracting joint index which equivalent to link index
    # print("Debug13::len(fingerTips_jointInfo)",len(fingerTips_jointInfo))
  
    for i in fingerTips_jointInfo:
      i=i[0]
      #print("Debug11 i",i)
      #print("Debug12::i['jointIndex']",i["jointIndex"])
      fingerTips_linkIndex.append(i["jointIndex"])  
    #print("Debug14::fingerTips_linkIndex",fingerTips_linkIndex)
 
    return fingerTips_linkIndex
    

  def get_wrist_index(self):
      wrist_index = []
      wrist_jointinfo = []
      #wrist_joint_name_bytes = [i.encode(encoding='UTF-8',errors='strict') for i in self.wrist_joint_name]
      for i in self.wrist_joint_name:
        wrist_jointinfo.append(self.jointInfo.searchBy(key="jointName",value = i))
      for i in wrist_jointinfo:
        i=i[0]
        wrist_index.append(i["jointIndex"])
      #print("wrist index",wrist_index)
      return wrist_index

  # def check_wrist_action_complete(self,motorCommands):
  #   wrist_state = []
  #   wrist_index = self.get_wrist_index()
  #   setcommand = []
  #   sumerror = 0
  #   for i in wrist_index:
  #     wrist_pos = p.getJointState(self.handId,i)[0]
  #     wrist_state.append(wrist_pos)
  #     setcommand.append(motorCommands[i])
  #     #sumerror += abs(setcommand[i]-wrist_state[i])
  #   print("wrist state",wrist_state)
  #   #print("error",sumerror)
  #   print("set",setcommand)
  #   if sumerror <= 1e-5:
  #     return True
  #   else:
  #     return False   
      
  def cameraSetup(self):
    #https://github.com/bulletphysics/bullet3/issues/1616
    width = 128
    height = 128
    
    fov = 60 # field of view
    aspect = width/height
    near = 0.02
    far =1
    endEffector_info = p.getLinkState(self.handId,self.kukaEndEffectorIndex,computeForwardKinematics=True)
    # print("endEffector_info",endEffector_info)
    
    endEffector_pos  = endEffector_info[4]
    endEffector_ori  = endEffector_info[5]

    # print("endEffector_pos",endEffector_pos)
    # print("endEffector_ori",endEffector_ori)
    endEffector_pos_Xoffset =0.
    endEffector_pos_Zoffset =-0.05

    endEffector_pos_InWorldPosition = endEffector_pos
    cameraEyePosition = endEffector_pos_InWorldPosition

    cameraEyePosition_ = [endEffector_pos[0]+endEffector_pos_Xoffset,endEffector_pos[1],endEffector_pos[2]+endEffector_pos_Zoffset]
    rot_matrix = p.getMatrixFromQuaternion(endEffector_ori)
    rot_matrix = np.array(rot_matrix).reshape(3, 3)

    # Initial vectors
    init_camera_vector = (0, 0, 1) # z-axis
    init_up_vector = (0, 1, 0) # y-axis
    # Rotated vectors
    camera_vector = rot_matrix.dot(init_camera_vector)
    up_vector = rot_matrix.dot(init_up_vector)
    
    cameraEyePosition = endEffector_pos_InWorldPosition
    dist_cameraTargetPosition = -0.02

    view_matrix =  p.computeViewMatrix(cameraEyePosition_, cameraEyePosition + 0.1 * camera_vector, up_vector)

    projection_matrix = p.computeProjectionMatrixFOV(fov, aspect, near, far)

    # Get depth values using the OpenGL renderer
   
    images = p.getCameraImage(width, height, view_matrix, projection_matrix, shadow=True,renderer=p.ER_BULLET_HARDWARE_OPENGL)
    rgb_opengl= np.reshape(images[2], (height, width, 4))*1./255.
    depth_buffer_opengl = np.reshape(images[3], [width, height])
    depth_opengl = far * near / (far - (far - near) * depth_buffer_opengl)
    seg_opengl = np.reshape(images[4], [width, height])*1./255.
    time.sleep(1)
  
  def reset(self):
    
    dirpath = os.getcwd()
    # self._hand = p.loadSDF(dirpath+"/gym_test/envs/shadow_hand_vijay/model.sdf")
    hand_path = resource_filename(__name__,"./shadow_hand_vijay/model.sdf")
    self._hand = p.loadSDF(hand_path)
    self.handId = self._hand[0]
  
    #self.cameraSetup()
    # reset joints to the middle valu 
    
    #p.resetBasePositionAndOrientation(self.handId,[-0.100000,0.000000,0.070000],[0.000000,0.000000,0.000000,1.000000])
    p.resetBasePositionAndOrientation(self.handId, [0.000, 0.000, 0.000],[0.7071, 0.000000, 0.000000, -0.7071])
    self.jointInfo.get_infoForAll_joints(self._hand)
    self.numJoints = p.getNumJoints(self.handId)
    self.num_Active_joint = self.jointInfo.getNumberOfActiveJoints()
    #print(" self.num_Active_join:::", self.num_Active_joint)
    self.indexOf_activeJoints  = self.jointInfo.getIndexOfActiveJoints()
    #reseting to the middle
 
    #resetting to the upper limit

    # print("Debug::self.indexOf_activeJoints",self.indexOf_activeJoints)
    
    
    self.trayUid = p.loadURDF(os.path.join(self.urdfRootPath,"tray/tray.urdf"), 0.640000,0.075000,-0.190000,0.000000,0.000000,1.000000,0.000000)
    self.endEffectorPos = [0.537,0.0,0.5]
    self.endEffectorAngle = 0
    
    
    self.motorNames = []
    self.motorIndices = []
    
    for i in range (self.numJoints):
      jointInfo = p.getJointInfo(self.handId,i)
      qIndex = jointInfo[3]
      if qIndex > -1:
        #print("motorname")
        #print(jointInfo[1])
        self.motorNames.append(str(jointInfo[1]))
        self.motorIndices.append(i)
 
  def getActionDimension(self):
    numOf_activeJoints = self.jointInfo.numberOfActiveJoints()
    return numOf_activeJoints #position x,y,z and roll/pitch/yaw euler angles of end effector

  def getObservationDimension(self):
    return len(self.getObservation())

  def getObservation_joint(self,format="list"):
    
    indexOfActiveJoints = self.jointInfo.getIndexOfActiveJoints()
    jointsInfo = self.jointInfo.getActiveJointsInfo()

    jointsStates = []
    joints_state = {} #key:joint ,value = joint state 
    
    for i in range(len(indexOfActiveJoints)):
      jointName  = jointsInfo[i]["jointName"]
      jointIndex = indexOfActiveJoints[i]
      jointState = p.getJointState(self._hand[0],jointIndex)
      joints_state[jointName] = jointState[0]
      jointsStates.append(jointState[0])

    if format == "dictinary":
      return joints_state
    else:
      return jointsStates
    

  def getObservation(self):
    #self.cameraSetup()
    state_dict = {}
    observation = []
    
    fingerTipIndexs = self.get_fingerTips_linkIndex()
    #print("Debug::fingerTipIndexs",fingerTipIndexs)
    counter = 0 
    #getting fingers tip position and orientation
    for index in fingerTipIndexs:
      state = p.getLinkState(self.handId,index)#mamad:returns endeffector info position orientation
      pos = state[0] #mamad: linkWorldPosition
      orn = state[1] #mamad: linkWorldOrientation
      state_dict[self.fingerTip_joint_name[counter]] = {"pos":pos,"orn":orn}                                          
      counter +=1
    
    #print("Debug::state_dict",state_dict)

    for finger in self.fingerTip_joint_name:
      euler = p.getEulerFromQuaternion(state_dict[finger]["orn"])
      pos   = state_dict[finger]["pos"]  
      observation.extend(list(pos))
      # observation.extend(list(euler))
    #print("Debug::observation",observation)
    #print("Debug::len(observation)",len(observation))
    return observation

  def applyAction(self, motorCommands):
      #The actions are going to Active joint values.
      #gettting current state of Avtive joints before applying actions This is different that the state we get in getObservation
      joint_state = [] #current joint postion
      new_joint_pos = [0]*self.jointInfo.getNumberOfActiveJoints() # new joint position
      for jointIndex in self.indexOf_activeJoints:
        joint_pos = p.getJointState(self.handId,jointIndex)[0]
        joint_state.append(joint_pos)
      # print("Debug::joint_state",joint_state)
      #making sure the joint values suggested by agent does not exceed joint limit
      #design question: should i give negative reward to agent for suggesting a joint value outside joint limit
      counter = 0
      limit=[]
      commedact=[] 
      for jointIndex in self.indexOf_activeJoints:
        jointinfo = self.jointInfo.searchBy("jointIndex",jointIndex)[0]
        joint_ll = jointinfo["jointLowerLimit"]
        joint_ul = jointinfo["jointUpperLimit"]
        # print("jointIndex",jointIndex)
        # if (jointIndex in cc):
        #   print("count",jointIndex,"jointName",self.jointInfo.searchBy("jointIndex",jointIndex)[0]["jointName"])
        # print("joint_ll",joint_ll)
        # print("joint_ul",joint_ul)
        limit.append([joint_ll,joint_ul])
        if (motorCommands[counter]>joint_ll or motorCommands[counter]<joint_ll):
          commedact.append([motorCommands[counter],True])
        else:
          commedact.append([motorCommands[counter],False])
        # print("---motorCommands:::",motorCommands[counter])
        # print("---motorCommands:::",motorCommands[counter])
        #   print("motorCommands[counter]<=joint_ul ",motorCommands[counter]<=joint_ul )
        #   print("motorCommands[counter]>=joint_ll",motorCommands[counter]>=joint_ll)
        if motorCommands[counter]<=joint_ul and motorCommands[counter]>=joint_ll:
          new_joint_pos[counter] = motorCommands[counter]
          counter +=1
      #Applying new_joint_pos to hand
      counter = 0
      # print("\n\n")
      # print("action",commedact)
      # print("limits",limit)
      # print("\n\n")
      for jointIndex in self.indexOf_activeJoints:
          #print("joint index",self.jointInfo.searchBy("jointIndex",jointIndex)[0]["jointName"],"new_joint_pos::",new_joint_pos[counter],"  motorCommands:::",motorCommands[counter])
          p.setJointMotorControl2(bodyUniqueId=self.handId,jointIndex=jointIndex,
                                  controlMode=p.POSITION_CONTROL,
                                  targetPosition=motorCommands[counter],
                                  targetVelocity=0,force=self.maxForce,
                                  maxVelocity=self.maxVelocity, positionGain=0.3,velocityGain=1)
          counter +=1

      # joint_indexx = self.indexOf_activeJoints[5]
      # print("joint index",self.jointInfo.searchBy("jointIndex",joint_indexx)[0])
      # p.setJointMotorControl2(bodyUniqueId=self.handId,jointIndex=joint_indexx,
      #                             controlMode=p.POSITION_CONTROL,
      #                             targetPosition=1
      #                             )
      


      """
      Todo: 
      []Fix apply function so all the fingers move
      []The only code here should be for hand move the camera code to handGymEnv

      Deadline: 18 jun 19

      """