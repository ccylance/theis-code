import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
print ("current_dir=" + currentdir)
os.sys.path.insert(0,currentdir)

import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import time
import pybullet as p
import hand
import random
import pybullet_data
from pkg_resources import parse_version,resource_string,resource_filename
import datetime
import scipy.misc
largeValObservation = 100

RENDER_HEIGHT = 720
RENDER_WIDTH = 960

class HandGymEnv(gym.Env):
  metadata = {
      'render.modes': ['human', 'rgb_array'],
      'video.frames_per_second' : 50
  }

  def __init__(self,
               urdfRoot=pybullet_data.getDataPath(),
               actionRepeat=1,
               isEnableSelfCollision=True,
               renders=True,
               isDiscrete=False,
               isLocalspace=False,
               maxSteps = 2000):
    self.rpy_target = []
    # [2.192842715221812, 5.836050621566519e-06, 0.0588609766793831]
    self._isDiscrete = isDiscrete
    self._timeStep = 1./500.
    self._urdfRoot = urdfRoot
    self._actionRepeat = actionRepeat
    self._isEnableSelfCollision = isEnableSelfCollision
    self.isLocalspace = isLocalspace
    self._observation = []
    self._envStepCounter = 0
    self._renders = renders
    self.step_count = 0
    self._maxSteps = maxSteps
    self.terminated = False
    self._cam_dist = 0.4
    self._cam_yaw = 180
    self._cam_pitch = -40
    self.action = []
    self.obs_old = None
    self._p = p
    self.eventFlags = False
    # self.unity = unity_test()
    self.reward = 0
    self.dt = 0
    self.dtPlus1 = 0
    self.action_old = None
    self.reward_count = 0
    self.reward_count_finish=0
    high_obs=np.array([0.3,0.25,0.25,0.3,0.25,0.25,0.3,0.25,0.25,0.3,0.25,0.25,0.3,0.25,0.25,0.5,0.5,0.5,2*3.1415926])
    low_obs=np.array([-0.3,-0.25,-0.25,-0.3,-0.25,-0.25,-0.3,-0.25,-0.25,-0.3,-0.25,-0.25,-0.3,-0.25,-0.25,-0.5,-0.5,-0.5,0])
    high_joint_limit=np.array([0.174533,0.488692,0.349066,1.5707963267948966,1.5707963267948966,0.349066,1.5707963267948966,
    1.5707963267948966,0.349066,1.5707963267948966,1.5707963267948966,1.5707963267948966,0.785398,0.349066,1.5707963267948966,
    1.5707963267948966,1.5707963267948966,1.0472,1.22173,0.20944,0.698132,1.5708])
    low_joint_limit=np.array([-0.523599,-0.698132,-0.349066,0,0,-0.349066,0,0,-0.349066,0,0,0,0,-0.349066,0,0,0,-1.0472,0,-0.20944,-0.698132,0])
    self.action_space = spaces.Box(low_joint_limit, high_joint_limit, dtype=np.float32)
    self.observation_space = spaces.Box(low_obs, high_obs, dtype=np.float32)
    if self._renders:
      cid = p.connect(p.SHARED_MEMORY)
      if (cid<0):
         cid = p.connect(p.GUI)
      p.resetDebugVisualizerCamera(1.3,180,-41,[0.52,-0.2,-0.33])
    else:
      p.connect(p.DIRECT)
    #timinglog = p.startStateLogging(p.STATE_LOGGING_PROFILE_TIMINGS, "kukaTimings.json")
    self._seed()
    self.reset()
    observationDim = len(self.getExtendedObservation())
    #print("observationDim")
    #print(observationDim)

    observation_high = np.array([largeValObservation] * observationDim)
    if (self._isDiscrete):
      self.action_space = spaces.Discrete(7)
    else:
       action_dim = self._hand.num_Active_joint
       self._action_bound = 1
       action_high = np.array([self._action_bound] * action_dim)
       self.action_space = spaces.Box(-action_high, action_high)
    self.observation_space = spaces.Box(-observation_high, observation_high)
    self.viewer = None

  def reset(self):

    import os
 
    dirpath = os.getcwd()
    # print("current directory is : " + dirpath)
    #print("KukaGymEnv _reset")
    self.terminated = False
    p.resetSimulation()
    # texUid = p.loadTexture(dirpath+"/gym_test/envs/cube_new/aaa.png")
    # cube_objects = p.loadSDF(dirpath+"/gym_test/envs/cube_new/model.sdf")
    cube_texture_path = resource_filename(__name__,"cube_new/aaa.png")
    cube_path = resource_filename(__name__,"cube_new/model.sdf")
    texUid = p.loadTexture(cube_texture_path)
    cube_objects = p.loadSDF(cube_path)
    # texUid = p.loadTexture("./cube_new/aaa.png")
    # cube_objects = p.loadSDF("./cube_new/model.sdf")
    self.cubeId = cube_objects[0]
    p.changeVisualShape(self.cubeId, -1, rgbaColor=[1, 1, 1, 1])
    p.changeVisualShape(self.cubeId, -1, textureUniqueId=texUid)
    p.resetBasePositionAndOrientation(self.cubeId, [0, 0.34, 0.07],[0.7071, 0.0000, 0.0000, 0.7071])
    # p.setPhysicsEngineParameter(numSolverIterations=150)
    p.setTimeStep(self._timeStep)
    p.loadURDF(os.path.join(self._urdfRoot,"plane.urdf"),[0,0,-1])
    
    cube_objects
    xpos = 0.58
    ypos = 0.04
    ang = 3.14*0.5
    orn = p.getQuaternionFromEuler([0,0,ang])
   

    self.cube = cube_objects
    #This where robot is reset i should modefy this to adjust height of end-effector
    p.setGravity(0,0,-10)
    self._hand = hand.Hand(urdfRootPath=self._urdfRoot, timeStep=self._timeStep)
    self._envStepCounter = 0
    p.stepSimulation()
    self._observation = self.getExtendedObservation()
    return np.array([self._observation])

  def __del__(self):
    p.disconnect()

  def _seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)
    return [seed]

  def getExtendedObservation(self):
    """
    returns postion of figer tips relative to the workd or the camera frame - top camera
    """
    observation = []
    # camera frame
    self._observation = self._hand.getObservation()
    if (self.isLocalspace): 
      # print("\n\n\n\n")
      # print("obhbbb",self._observation)
      # print("\n\n\n\n")
      camPos=[0,2,0]
      camOrn = p.getQuaternionFromEuler([0,90,0])
      blockPos = p.getLinkState(self.cubeId ,-1)[0]
      blockOrn = p.getLinkState(self.cubeId ,-1)[1]
      invcamPos,invcamOrn = p.invertTransform(camPos,camOrn)
      camMat = p.getMatrixFromQuaternion(camOrn)

      # block in camera frame
      blockPosInCam,blockOrnInCam = p.multiplyTransforms(invcamPos,invcamOrn,blockPos,blockOrn)
      blockEulerInCam = p.getEulerFromQuaternion(blockOrnInCam)

      # fingers pos in camera frmae
      for fingerTip in self._hand.fingerTip_joint_name:
        fingerPosInCam,fingerOrnInCam = p.multiplyTransforms(invcamPos,invcamOrn, self._observation[fingerTip]["pos"],self._observation[fingerTip]["orn"])
        observation.extend(list(fingerPosInCam))  
      
      observation.extend(list(blockPosInCam))
      observation.extend(list(blockEulerInCam))
    # simulation world frame
    else: 
      blockPos,blockOrn = p.getBasePositionAndOrientation(self.cubeId)
      observation.extend(list(self._observation))
      observation.extend(list(blockPos))
      observation.extend(list(blockOrn))
      # print("blockOrn",p.getEulerFromQuaternion(blockOrn))
    return observation

  def step(self, action,target):
    # print("HandGymEnv::_step")
    # print("HandGymEnv::_step::action[0]",action[0])
    # print("\n\n")
    actions = action
    self.step_count +=1

    self.action = actions
    return self.step2(actions,target)

  def step2(self, action,target):
    for i in range(self._actionRepeat):
  
      
      if self._termination():
        break
      self._hand.applyAction(action)
      p.stepSimulation()
      self._envStepCounter = self._envStepCounter+1
    if self._renders:
      time.sleep(self._timeStep)


    self._observation = self.getExtendedObservation()
    self.rpy_target=target
    
    done = self._termination()
    reward = self._reward()
    # print("check_if_self_collision_has_happend",self.check_if_self_collision_has_happend())
    # self.render()
    
    return np.array([self._observation]), reward, done, {}

  def render(self, mode="rgb_array", close=False):
    if mode != "rgb_array":
      return np.array([])
    
    base_pos,orn = self._p.getBasePositionAndOrientation(self._hand.handId)
    view_matrixup = self._p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=[0,0,self._cam_dist],
        distance=self._cam_dist,
        #yaw=self._cam_yaw,
        yaw = 0,
        #pitch=self._cam_pitch,
        pitch = -90,
        roll=0,
        upAxisIndex=2)
    # view_matrixleft = self._p.computeViewMatrixFromYawPitchRoll(
    #     cameraTargetPosition=[0,self._cam_dist,0],
    #     distance=self._cam_dist,
    #     #yaw=self._cam_yaw,
    #     yaw = 90,
    #     #pitch=self._cam_pitch,
    #     pitch = 0,
    #     roll=0,
    #     upAxisIndex=2)    
    # view_matrixright = self._p.computeViewMatrixFromYawPitchRoll(
    #     cameraTargetPosition=[0,self._cam_dist,0],
    #     distance=self._cam_dist,
    #     #yaw=self._cam_yaw,
    #     yaw = -90,
    #     #pitch=self._cam_pitch,
    #     pitch = 0,
    #     roll=0,
    #     upAxisIndex=2)  

    proj_matrix = self._p.computeProjectionMatrixFOV(
        fov=60, aspect=float(RENDER_WIDTH)/RENDER_HEIGHT,
        nearVal=0.1, farVal=100.0)
    (_, _, pxup, _, _) = self._p.getCameraImage(
        width=RENDER_WIDTH, height=RENDER_HEIGHT, viewMatrix=view_matrixup,
        projectionMatrix=proj_matrix, renderer=self._p.ER_BULLET_HARDWARE_OPENGL)
        #renderer=self._p.ER_TINY_RENDERER)
  
    # (_, _, pxleft, _, _) = self._p.getCameraImage(
    #     width=RENDER_WIDTH, height=RENDER_HEIGHT, viewMatrix=view_matrixleft,
    #     projectionMatrix=proj_matrix, renderer=self._p.ER_BULLET_HARDWARE_OPENGL)

    # (_, _, pxright, _, _) = self._p.getCameraImage(
    #     width=RENDER_WIDTH, height=RENDER_HEIGHT, viewMatrix=view_matrixright,
    #     projectionMatrix=proj_matrix, renderer=self._p.ER_BULLET_HARDWARE_OPENGL)

    rgb_arrayup = np.array(pxup, dtype=np.uint8)
    rgb_arrayup = np.reshape(rgb_arrayup, (RENDER_HEIGHT, RENDER_WIDTH, 4))	
    rgb_arrayup = rgb_arrayup[:, :, :3]

    # rgb_arrayleft = np.array(pxleft, dtype=np.uint8)
    # rgb_arrayleft = np.reshape(rgb_arrayleft, (RENDER_HEIGHT, RENDER_WIDTH, 4))	
    # rgb_arrayleft = rgb_arrayleft[:, :, :3]

    # rgb_arrayright = np.array(pxright, dtype=np.uint8)
    # rgb_arrayright= np.reshape(rgb_arrayright, (RENDER_HEIGHT, RENDER_WIDTH, 4))	
    # rgb_arrayright = rgb_arrayright[:, :, :3]

    
    # return rgb_arrayup,rgb_arrayleft,rgb_arrayright
    return rgb_arrayup
  def _termination(self):
    """
    terninate if max reward is reached. solved!
    terminates if max steps is reached
    terminates if self collision occures
    """

    threshold = 0.001
    if (self.terminated or self._envStepCounter>self._maxSteps):
      return True


    return False


  def _reward(self):
    """
    rt = dt+dt+1
    """
    # need to be adjusted
    d_cube_orn_threshold = 0.3
   
    # difference between current angle and target angle
    cube_pos,cube_orn = p.getBasePositionAndOrientation(self.cubeId)
    cube_orn = p.getEulerFromQuaternion(cube_orn)
    d_cube_orn = []
    for i in range(len(cube_orn)):
      d_theta =(cube_orn[i]-self.rpy_target[i])**2
      d_cube_orn.append(d_theta)

    self.dtPlus1 = math.sqrt(sum(d_cube_orn))
    rt = self.dt+self.dtPlus1
    self.dt = self.dtPlus1

    self.reward =-1* rt

    # target achieved
    if self.dtPlus1 < d_cube_orn_threshold:
      self.reward +=5
      self.terminated = True

    # object falling
    if cube_pos[2] <= p.getLinkState(self._hand.handId,2)[0][2]:
      self.reward -=20
      self.terminated = True
    

    return self.reward


  def render_sim(self):
    cid = p.connect(p.SHARED_MEMORY)
    if (cid<0):
      cid = p.connect(p.GUI)
  if parse_version(gym.__version__)>=parse_version('0.9.6'):
    render = render
    reset  = reset
    seed   = _seed
    step   = step

  def saveimage(self,rgbup,rgbleft,rgbright):
    print("save--------------")
    today = datetime.datetime.now()
    #saving the image with the current time
    # scipy.misc.imsave('shadow_hand_vijay_cam/imageup/imageup%s.jpg'%today.isoformat(), rgbup)
    # scipy.misc.imsave('shadow_hand_vijay_cam/imageleft/imageleft%s.jpg'%today.isoformat(), rgbleft)
    # scipy.misc.imsave('shadow_hand_vijay_cam/imageright/upimageright%s.jpg'%today.isoformat(), rgbright)
  # untility

  def check_if_self_collision_has_happend(self):
    jointInfo = self._hand.jointInfo

    active_joints_info  = jointInfo.getActiveJointsInfo()
    num_active_joints = jointInfo.getNumberOfActiveJoints()
    index_of_actvie_joints = [active_joints_info[i]["jointIndex"] for i in range(num_active_joints)]

    contact_set = []
    child_check = []
    for index_1 in index_of_actvie_joints:
      for index_2 in index_of_actvie_joints:
        if index_1 == index_2:
          continue
          
        contact=p.getClosestPoints(self._hand.handId,self._hand.handId,-0.01,index_1,index_2)

        if len(contact) !=0:
          link_one_name = jointInfo.searchBy("jointIndex",contact[0][3])[0]["linkName"]
          link_two_name = jointInfo.searchBy("jointIndex",contact[0][4])[0]["linkName"]
          contact_set.append([contact[0][3],contact[0][4]])
    new_contact = []
    for i in contact_set:
     
      if not i in new_contact:
        new_contact.append(i)
   
    parent_set = []
   
    for i in range(len(new_contact)):
      
      if new_contact[i][0] - new_contact[i][1] == 1:
        parent_set.append(new_contact[i])
 
    child_set = []
    for i in range(len(new_contact)):
      if new_contact[i][1] - new_contact[i][0] == 1:
        child_set.append(new_contact[i])
  

    parent_check = []
    for i in new_contact:
      for j in parent_set:
        if i == j:
          break
      else:
        parent_check.append(i)
   
    child_check = []
    for i in parent_check:
      for j in child_set:
        if i == j:
          break
      else:
        child_check.append(i)
    check_flip=[]
    for i in range(len(child_check)):
      index_1=child_check[i][0]
      index_2=child_check[i][1]
      for j in range(i,len(child_check)):
        if i == j:
          continue
        if index_1 == child_check[j][1] and index_2 ==  child_check[j][0]:
          check_flip.append(j)
    new_check=[]
    sort=np.argsort(check_flip)
    for i in range(len(check_flip)):
      new_check.append(check_flip[sort[i]])
    for i in range(len(check_flip)):
      del child_check[new_check[i]-i]

    collision_result = []
    for i in range (len(child_check)):
      index_collision_set_1=child_check[i][0]
      index_collision_set_2=child_check[i][1]
      for j in range(num_active_joints):
        if index_collision_set_1 == active_joints_info[j]["jointIndex"]:
          index_collision_set_1_result = j
        if index_collision_set_2 == active_joints_info[j]["jointIndex"]:
          index_collision_set_2_result = j	

      collision_result.append([active_joints_info[index_collision_set_1_result]["linkName"],\
      active_joints_info[index_collision_set_2_result]["linkName"]])
    # print(" hand self coliision -------",child_check)
    # print(" hand self coliision -------",collision_result)
    # print("\n")
    if len(collision_result)>1:
      return True
    else:
      return False 

  
    
class Demo():

  def __init__(self):
    self.env = HandGymEnv(renders=True, isDiscrete=False)
    self.env.reset()
    self.setp_counter = 0
   
  def get_commandIndexForpart(self,part):
    # stroing index of command in motor command 
    joint_names_Base = {}
    joint_names_TH   = {}
    joint_names_FF   = {}
    joint_names_MF   = {}
    joint_names_RF   = {}
    joint_names_LF   = {}

    indexOfActiveJoints = self.env._hand.jointInfo.getIndexOfActiveJoints()
    jointsInfo = self.env._hand.jointInfo.getActiveJointsInfo()

    # print("indexOfActiveJoints::",indexOfActiveJoints)
    # getting position of specifc command in motorCommand array
    for jointInfo in jointsInfo:
      joint_name = jointInfo["jointName"]
      jointIndex = jointInfo["jointIndex"]
      # print("jointIndex::",jointIndex,"  jointName::",joint_name)
      jointll  = jointInfo["jointLowerLimit"]
      jointul  = jointInfo["jointUpperLimit"]
      if "WR" in joint_name:
        joint_names_Base[joint_name] = {"commandIndex":indexOfActiveJoints.index(jointIndex),
                                        "jointIndex":jointIndex,
                                        "jointll":jointll,
                                        "jointul":jointul
                                        }
      if "TH" in joint_name:
        joint_names_TH[joint_name]   = {"commandIndex":indexOfActiveJoints.index(jointIndex),
                                        "jointIndex":jointIndex,
                                        "jointll":jointll,
                                        "jointul":jointul
                                        }
      if "FF" in joint_name:
        joint_names_FF[joint_name]   = {"commandIndex":indexOfActiveJoints.index(jointIndex),
                                        "jointIndex":jointIndex,
                                        "jointll":jointll,
                                        "jointul":jointul
                                        }
      if "MF" in joint_name:
        joint_names_MF[joint_name]   = {"commandIndex":indexOfActiveJoints.index(jointIndex),
                                        "jointIndex":jointIndex,
                                        "jointll":jointll,
                                        "jointul":jointul
                                        }
      if "RF" in joint_name:
        joint_names_RF[joint_name]   = {"commandIndex":indexOfActiveJoints.index(jointIndex),
                                        "jointIndex":jointIndex,
                                        "jointll":jointll,
                                        "jointul":jointul
                                        }
      if "LF" in joint_name:
        joint_names_LF[joint_name]   = {"commandIndex":indexOfActiveJoints.index(jointIndex),
                                        "jointIndex":jointIndex,
                                        "jointll":jointll,
                                        "jointul":jointul
                                        }
      

    if part =="FF":
      return joint_names_FF
    elif part =="MF":
      return joint_names_MF
    elif part =="RF":
      return joint_names_RF
    elif part =="LF":
      return joint_names_LF
    elif part=="Base":
      return joint_names_Base

  def get_active_joints_name(self):
    jointsInfo = self.env._hand.jointInfo.getActiveJointsInfo()
    jointsNames = [jointInfo["jointName"] for jointInfo in jointsInfo]
    return jointsNames
  
  
  def move_finger_y_middle(self,fingerName="FF",joint_name ="J3_FF"):
    print("Demo::move_finger_y")
    jointsStates = self.env._hand.getObservation_joint()
    motorCommand = jointsStates

    index_of_commandsIn_motorCommand = self.get_commandIndexForpart(fingerName)

    # print("index_of_motorCommand",index_of_commandsIn_motorCommand)

    # incremetally increasing joint value
    # increment = (jointul+jointll)/100

    if self.setp_counter%10 == 0:
      for jointName in index_of_commandsIn_motorCommand:
        
        if jointName == joint_name:
          print("jointName::",jointName)
          jointll = index_of_commandsIn_motorCommand[jointName]["jointll"]
          jointul = index_of_commandsIn_motorCommand[jointName]["jointul"]
          commandIndex = index_of_commandsIn_motorCommand[jointName]["commandIndex"] 
          motorCommand[commandIndex] =(jointll+jointul)/2
          print("motorCommand[commandIndex]::",motorCommand[commandIndex])
          print("jointll::",jointll)
          print("jointul::",jointul)
          print(" motorCommand[commandIndex]<jointul and  motorCommand[commandIndex]>jointll", motorCommand[commandIndex]<jointul and  motorCommand[commandIndex]>jointll)
          if motorCommand[commandIndex]<jointul and  motorCommand[commandIndex]>jointll:
            print("increment added")
            motorCommand[commandIndex] = motorCommand[commandIndex]
          else:
            print("reseting to lower limit")
            motorCommand[commandIndex] = jointll

    self.setp_counter +=1
    # print("Demo::move_finger_y::motorCommand",motorCommand)
    print("\n\n")
    self.env.step([motorCommand])

  def move_finger_y(self,fingerName="FF",joint_name ="J4_FF",movement_direction = "positive"):
    print("Demo::move_finger_y")
    jointsStates = self.env._hand.getObservation_joint()
    motorCommand = jointsStates

    index_of_commandsIn_motorCommand = self.get_commandIndexForpart(fingerName)

    # print("index_of_motorCommand",index_of_commandsIn_motorCommand)

    # incremetally increasing joint value
    # increment = (jointul+jointll)/100

    if self.setp_counter%10 == 0:
      for jointName in index_of_commandsIn_motorCommand:
        
        if jointName == joint_name:
          print("jointName::",jointName)
          jointll = index_of_commandsIn_motorCommand[jointName]["jointll"]
          jointul = index_of_commandsIn_motorCommand[jointName]["jointul"]
          if(abs(jointll) ==abs(jointul)):
            increment = abs(jointul)/10
          else:
            increment = (abs(jointll)-abs(jointul))/10
            if increment <0 and movement_direction =="positive":
              increment = abs(increment)
            elif increment>0 and  movement_direction =="negative":
              increment = -1*increment
          
          commandIndex = index_of_commandsIn_motorCommand[jointName]["commandIndex"] 
          motorCommand[commandIndex] = jointsStates[commandIndex]+increment
          print("motorCommand[commandIndex]::",motorCommand[commandIndex])
          print("jointll::",jointll)
          print("jointul::",jointul)
          print(" motorCommand[commandIndex]<jointul and  motorCommand[commandIndex]>jointll", motorCommand[commandIndex]<jointul and  motorCommand[commandIndex]>jointll)
          if motorCommand[commandIndex]<jointul and  motorCommand[commandIndex]>jointll:
            print("increment added")
            motorCommand[commandIndex] = motorCommand[commandIndex]
          else:
            print("reseting to lower limit")
            motorCommand[commandIndex] = jointll

    self.setp_counter +=1
    # print("Demo::move_finger_y::motorCommand",motorCommand)
    print("\n\n")
    self.env.step([motorCommand])

  def move_joint(self,joint_name):

    indexOfActiveJoints = self.env._hand.jointInfo.getIndexOfActiveJoints()

    jointsStates = self.env._hand.getObservation_joint()
    #print("len(jointsStates)",len(jointsStates))
    motorCommand = jointsStates

    jointInfo = self.env._hand.jointInfo.searchBy(key="jointName",value=joint_name)[0]
    # print(jointInfo)
    jointll = jointInfo["jointLowerLimit"]
    jointul = jointInfo["jointUpperLimit"]
    jointIndex = jointInfo["jointIndex"]
    commandIndex_in_motorCommand = indexOfActiveJoints.index(jointIndex)
    # print("joint`index:: ",jointIndex,"   commandIndex::",commandIndex_in_motorCommand)
    # print("jointsStates[commandIndex_in_motorCommand]::",jointsStates[commandIndex_in_motorCommand])
    # print("jointll::",jointll)
    # print("jointul::",jointul)
    increment = (abs(jointll)-abs(jointul))/10

    # motorCommand[commandIndex_in_motorCommand] = jointsStates[commandIndex_in_motorCommand]+increment

    motorCommand[commandIndex_in_motorCommand] = jointul
    #print("motorCommand[commandIndex_in_motorCommand]",motorCommand[commandIndex_in_motorCommand])
    self.env.step([motorCommand])


  def debug(self):
    pass
    #self.env.reset()
    """
    jointIndex = self.env._hand.jointInfo.getIndexOfActiveJoints()
    motorCommands = [i for i in range(len(jointIndex))]
    for index in jointIndex:
      jointInfo = self.env._hand.jointInfo.searchBy(key="jointIndex",value=index)[0]
      jointName = jointInfo["jointName"]
      print(jointName," ",index)
    print("env::jointIndex  ",jointIndex)
    print("env::motorCommands  ",motorCommands)
    print("len(jointIndex)",len(jointIndex))
    print("len(motorCommands)",len(motorCommands))

    self.env.step([motorCommands])
    """
    

  def move_finger_x(self,fingerName="FF"):
    pass


  """ 
      -Todo:
      [*]Add three cameras to the env. The cameras should have the same location as the ones in Unity (Talk to Cindy about this)
       Deadline :18jun19
      [*]Delete the code in reward function and replicate the reward in open ai paper.
      [*] Add a cube to the Env
      [] visually check if hand works as expected by changing joint values gradually (mamad)
      [] save images
      [] add collision to the reward terminate when collision happens
      []fix  model.sdf.erb
      """