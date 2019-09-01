require 'yaml'

class Base
    #constants
    M_PI = 3.1415926535897931
    attr_accessor :forearm,:wrist,:palm,:mesh_c,:mesh_v,:model
    def initialize(sim_env,load_setting)

        @forearm ={  # object pose
                "link_name"       =>"forearm",
                'pose'=> [0,0,0,
                            0,0,0],
                # inetia 
                'pose_i'=> [0,0,0.09,
                                 0,0,0],                   
                'mass'=>3.0,
                'ixx'=> 0.0138 ,'ixy'=> 0     ,'ixz'=> 0 ,
                'iyy'=> 0.0138 ,'iyz'=> 0     ,'izz'=> 0.00744 ,
                #color
                "ambient"=>  [0,0,0,1],
                "diffuse"=>  [0,0,0,1],
                "specular"=> [0,0,0,1],
                "emissive"=> [0,0,0,0]
            }  
        @wrist = {   # object pose
                "link_name"       =>"wrist",
                'pose'=> [0, -0.01,0.213,
                0,0,0],
                # inetia 
                'pose_i'=> [0,0,0.029,
                                0,0,0],                   
                'mass'=> 0.1,
                'ixx'=> 0.000035 ,'ixy'=> 0     ,'ixz'=> 0 ,
                'iyy'=> 0.000064 ,'iyz'=> 0     ,'izz'=> 0.0000438, 
                # joint
                "p_name"     =>"forearm",
                "j_name"     =>"WRJ2",
                "j_type"     =>"revolute",
                "j_axis"     => [0,1,0],
                "j_limit"    => [-0.523599,0.174533], #[lower,upper]
                "j_effort"   => 10,
                "j_velocity" => 2.0,
                "j_damping"  => 0.1,
                "j_friction" => 0.0,
                "j_spring_reference" => 0.0,
                "j_spring_stiffness" => 0.0,
                #color
                "ambient"=>  [0,0,0,1],
                "diffuse"=>  [0,0,0,1],
                "specular"=> [0,0,0,1],
                "emissive"=> [0,0,0,0]

            }
        @palm =  {  # object pose
                "link_name"       =>"palm",
                'pose'=> [0.0,-0.01,0.247,
                          0,0,0],
                # inetia 
                'pose_i'=> [0,0,0.035,
                                0,0,0],                   
                'mass'=> 0.3,
                'ixx'=> 0.0003581 ,'ixy'=> 0     ,'ixz'=> 0 ,
                'iyy'=> 0.0005287 ,'iyz'=> 0     ,'izz'=> 0.000191,
                # joint
                "p_name"     =>"wrist",
                "j_name"     =>"WRJ1",
                "j_type"     =>"revolute",
                "j_axis"     => [1,0,0],
                "j_limit"    => [-0.698132,0.488692], #[lower,upper]
                "j_effort"   => 30,
                "j_velocity" => 2.0,
                "j_damping"  => 0.1,
                "j_friction" => 0.0,
                "j_spring_reference" => 0.0,
                "j_spring_stiffness" => 0.0,
                #color
                "ambient"=>  [0,0,0,1],
                "diffuse"=>  [0,0,0,1],
                "specular"=> [0,0,0,1],
                "emissive"=> [0,0,0,0]
        }  
        @model={"BASE"=>{"Links"=>{}}}
        def generate_model_forReset()
            link_list = [ @forearm,@wrist, @palm]
            i=0
            loop do
          
                @model["BASE"]["Links"][link_list[i]["link_name"]]={ 
        
                        "mass"      =>link_list[i]["mass"],

                        "color"     =>{
                            'ambient'  => link_list[i]["ambient"] ,                                              
                            'specular' => link_list[i]["specular"],
                            'emissive' => link_list[i]["emissive"],
                            'diffuse' => link_list[i]["diffuse"] 
                        },
                        "joint"     =>{

                            "damping"     =>link_list[i]["j_damping"],
                            "friction"      =>link_list[i]["j_friction"]

                        }
                    }
                if i == link_list.length-1
                 break
                end
                i +=1
          
           end
        
        end
        generate_model_forReset()
        def loadYamlAndChange()
            file = YAML.load_file("config.yml")
            #print file 
         
                
            #print finger
            links = file["BASE"]["Links"]
            links.each do |linkName,link|

                color = link["color"]
                mass  = link["mass"]
                if !(linkName == "forearm")
                    joint = link["joint"]
                end
                

                if linkName == "forearm"
                    #visual
                    @forearm["ambient"] = color["ambient"]
                    @forearm["diffuse"] = color["diffuse"]
                    @forearm["specular"] = color["specular"]
                    @forearm["emissive"] = color["emissive"]
                    #physical prop
                    @forearm["mass"] = mass

                elsif linkName== "wrist"
                    #visual
                    @wrist["ambient"] = color["ambient"]
                    @wrist["diffuse"] = color["diffuse"]
                    @wrist["specular"] = color["specular"]
                    @wrist["emissive"] = color["emissive"]
                    #physical prop
                    @wrist["mass"] = mass
                    @wrist["j_damping"] = joint["damping"]
                    @wrist["j_friction"] = joint["friction"]
                elsif linkName== "palm"
                    #visual
                    @palm["ambient"] = color["ambient"]
                    @palm["diffuse"] = color["diffuse"]
                    @palm["specular"] = color["specular"]
                    @palm["emissive"] = color["emissive"]
                    #physical prop
                    @palm["mass"] = mass
                    @palm["j_damping"] = joint["damping"]
                    @palm["j_friction"] = joint["friction"]

                end
            end
              
            
        end
        if load_setting == true
            loadYamlAndChange()
        end
        path_v ={
                "palm"   => "meshes/palm_mirror.dae",
                "forearm"   =>"meshes/forearm.dae",
                "wrist" => "meshes/wrist.dae",
            }
        path_c ={
                "palm"   => "meshes/palm_mirror.dae",
                "forearm"   =>"meshes/forearm.dae",
                "wrist" => "meshes/wrist.dae",  
            }
        @mesh_c={}
        @mesh_v={}
        
        if sim_env == "gazebo"
            #model://+default path
            path_c.each do |key,value|
              @mesh_c.merge!({key => "model://shadow_hand_vijay/"+value})
            end
            path_v.each do |key,value|
              @mesh_v.merge!({key => "model://shadow_hand_vijay/"+value})
            end

        elsif sim_env == "pybullet"
            #deafult path
            @mesh_v = path_v
            @mesh_c = path_c
        else
            puts "Fingers:   well life sucks!not everything is supported.A'nt life a bitch? "
        end


    end
end
class Finger
    #constants
    M_PI = 3.1415926535897931
    attr_accessor :knuckle,:proximal,:middle,:distal,:palm,:fingername,:mesh_c,:mesh_v,:model
    def initialize(sim_env,load_setting,palm,biase,fingername,sensor_type)
        @fingername = fingername
        @palm = palm
        @knuckle =  {  # object pose
                        "link_name"       =>"knuckle"+fingername,
                        'pose'=> [0.033+biase[0],
                                   -0.01+biase[1],
                                   0.342+biase[2],
                                   0,0,0],
                        # inetia 
                        'pose_i'=> [0,0,0,
                                    0,0,0],                   
                        'mass'=> 0.008,
                        'ixx'=> 0.00000026 ,'ixy'=> 0     ,'ixz'=> 0 ,
                        'iyy'=> 0.00000026 ,'iyz'=> 0     ,'izz'=>0.00000032,
                        #Link
                        "scale" =>[0.001,0.001,0.001],
                        # joint 
                        "name"       =>"knuckle",
                        "p_name"     =>"palm",
                        "j_name"     =>"J4_"+fingername,
                        "j_type"     =>"revolute",
                        "j_axis"     => [0,-1,0],
                        "j_limit"    => [-20/180*M_PI,20/180*M_PI], #[lower,upper]
                        "j_effort"   => 2.0,
                        "j_velocity" => 2.0,
                        "j_damping"  => 0.1,
                        "j_friction" => 0.0,
                        #color
                        "ambient"=>  [0,0,0,1],
                        "diffuse"=>  [0,0,0,1],
                        "specular"=> [0,0,0,1],
                        "emissive"=> [0,0,0,0]
            }
        @proximal = {  # object pose
                    "link_name"       =>"proximal"+fingername,
                    'pose'=> [0+knuckle['pose'][0],
                            0+knuckle['pose'][1],
                            0+knuckle['pose'][2],
                            0,0,0],
                    # inetia 
                    'pose_i'=> [0,0,0.0225,
                                    0,0,0],                   
                    'mass'=> 0.030,
                    'ixx'=> 0.0000098 ,'ixy'=> 0     ,'ixz'=> 0 ,
                    'iyy'=> 0.00001 ,'iyz'=> 0     ,'izz'=>0.0000018 ,
                    #Link
                    "scale" =>[0.001,0.001,0.001],
                     # joint 
                     "name"       =>"proximal",
                     "p_name"     =>"knuckle"+fingername,
                     "j_name"     =>"J3_"+fingername,
                     "j_type"     =>"revolute",
                     "j_axis"     => [1,0,0],
                     "j_limit"    => [0,M_PI/2], #[lower,upper]
                     "j_effort"   => 2.0,
                     "j_velocity" => 2.0,
                     "j_damping"  => 0.1,
                     "j_friction" => 0.0,
                     #color
                     "ambient"=>  [0,0,0,1],
                     "diffuse"=>  [0,0,0,1],
                     "specular"=> [0,0,0,1],
                     "emissive"=> [0,0,0,0]
                } 
        @middle =   {  # object pose
                    "link_name"       =>"middle"+fingername,
                    'pose'=> [0+proximal['pose'][0],
                            0+proximal['pose'][1],
                            0.045+proximal['pose'][2],
                            0,0,0],
                    # inetia 
                    'pose_i'=> [0,0,0.0125,
                                    0,0,0],                   
                    'mass'=> 0.017,
                    'ixx'=>0.0000026 ,'ixy'=> 0     ,'ixz'=> 0 ,
                    'iyy'=>0.0000027 ,'iyz'=> 0     ,'izz'=>0.00000087,
                    #Link
                    "scale" =>[0.001,0.001,0.001],
                    # joint 
                    "name"       =>"middle",
                    "p_name"     =>"proximal"+fingername,
                    "j_name"     =>"J2_"+fingername,
                    "j_type"     =>"revolute",
                    "j_axis"     => [1,0,0],
                    "j_limit"    => [0,M_PI/2], #[lower,upper]
                    "j_effort"   => 2.0,
                    "j_velocity" => 2.0,
                    "j_damping"  => 0.1,
                    "j_friction" => 0.0,
                    #color
                    "ambient"=>  [0,0,0,1],
                    "diffuse"=>  [0,0,0,1],
                    "specular"=> [0,0,0,1],
                    "emissive"=> [0,0,0,0] 
                }
        @distal =   {  # object pose
                "link_name"       =>"distal"+fingername,
                'pose'=> [0+middle['pose'][0],
                          0+middle['pose'][1],
                          0.025+middle['pose'][2],
                          0,0,0],
                # inetia 
                'pose_i'=> [0,0,0.012,
                                0,0,0],                   
                'mass'=> 0.012,
                'ixx'=>0.00000094 ,'ixy'=> 0     ,'ixz'=> 0 ,
                'iyy'=>0.0000011 ,'iyz'=> 0     ,'izz'=>0.00000053,
                #Link
                "scale" =>[0.001,0.001,0.001],
                # joint 
                "name"       =>"distal",
                "p_name"     =>"middle"+fingername,
                "j_name"     =>"J1_"+fingername,
                "j_type"     =>"revolute",
                "j_axis"     => [1,0,0],
                "j_limit"    => [0,M_PI/2], #[lower,upper]
                "j_effort"   => 2.0,
                "j_velocity" => 2.0,
                "j_damping"  => 0.1,
                "j_friction" => 0.0,
                #color
                "ambient"=>  [0,0,0.5,1],
                "diffuse"=>  [0,0,0.5,1],
                "specular"=> [0,0,0.5,1],
                "emissive"=> [0,0,0,0]
        }
        @model={@fingername=>{"Links"=>{}}}
        def generate_model_forReset()
            link_list = [@knuckle,@proximal,@middle,@distal]
            i=0
            loop do
                
                @model[@fingername]["Links"][link_list[i]["link_name"]]={ 
        
                    "mass"      =>link_list[i]["mass"],

                    "color"     =>{
                                    'ambient'  => link_list[i]["ambient"] ,                                             
                                    'specular' => link_list[i]["specular"],
                                    'emissive' => link_list[i]["emissive"],
                                    'diffuse' => link_list[i]["diffuse"] 
                    },
                    "joint"     =>{

                                    "damping"     =>link_list[i]["j_damping"],
                                    "friction"      =>link_list[i]["j_friction"]

                    }
                }
                if i == link_list.length-1
                 break
                end
                i +=1
            
           end
         
        end
        generate_model_forReset()
        def distal_sensor_distal(sensor_type)
            if sensor_type =="default"
                return  "meshes/F1.dae"   
            elsif sensor_type =="eli"
                return  "meshes/distal_ellipsoid_visual.dae" 
            elsif sensor_type =="bio"
                 @distal["pose"] = [@distal["pose"][0],@distal["pose"][1],@distal["pose"][2]-0.009,@distal["pose"][3],@distal["pose"][4],@distal["pose"][5]]
                @distal["scale"] = [0.0254,0.0254,0.0254]
                return  "meshes/biotac_decimated.dae"
            end
        end
        #biotac_finger_adapter.dae
        def distal_sensor_middle(sensor_type)
            if sensor_type =="default"
                return  "meshes/F2.dae"   
            elsif sensor_type =="eli"
                return  "meshes/F2.dae"
            elsif sensor_type =="bio"
                @middle["scale"] = [1,1,1]
                @middle["pose"] = [@middle["pose"][0],@middle["pose"][1],@middle["pose"][2],@middle["pose"][3],@middle["pose"][4],1.571]
                return  "meshes/biotac_finger_adapter.dae"
            end
        end
        def loadYamlAndChange()
            file = YAML.load_file("config.yml")
            #print file 
         
                
            #print finger
            links = file[ @fingername]["Links"]
            links.each do |linkName,link|

                color = link["color"]
                mass  = link["mass"] 
                joint = link["joint"]

                if linkName == "knuckle"
                    #visual
                    @knuckle["ambient"] = color["ambient"]
                    @knuckle["diffuse"] = color["diffuse"]
                    @knuckle["specular"] = color["specular"]
                    @knuckle["emissive"] = color["emissive"]
                    #physical prop
                    @knuckle["mass"] = mass
                    @knuckle["j_damping"] = joint["damping"]
                    @knuckle["j_friction"] = joint["friction"]
                elsif linkName== "proximal"
                    #visual
                    @proximal["ambient"] = color["ambient"]
                    @proximal["diffuse"] = color["diffuse"]
                    @proximal["specular"] = color["specular"]
                    @proximal["emissive"] = color["emissive"]
                    #physical prop
                    @proximal["mass"] = mass
                    @proximal["j_damping"] = joint["damping"]
                    @proximal["j_friction"] = joint["friction"]
                elsif linkName== "middle"
                    #visual
                    @middle["ambient"] = color["ambient"]
                    @middle["diffuse"] = color["diffuse"]
                    @middle["specular"] = color["specular"]
                    @middle["emissive"] = color["emissive"]
                    #physical prop
                    @middle["mass"] = mass
                    @middle["j_damping"] = joint["damping"]
                    @middle["j_friction"] = joint["friction"]
                elsif linkName== "distal"
                    #visual
                    @distal["ambient"] = color["ambient"]
                    @distal["diffuse"] = color["diffuse"]
                    @distal["specular"] = color["specular"]
                    @distal["emissive"] = color["emissive"]
                    #physical prop
                    @distal["mass"] = mass
                    @distal["j_damping"] = joint["damping"]
                    @distal["j_friction"] = joint["friction"]
                end
            end
              
            
        end
        if load_setting == true
            loadYamlAndChange()
        end
        path_v ={
                "distal"   => distal_sensor_distal(sensor_type),
                "middle"   => distal_sensor_middle(sensor_type),
                "proximal" => "meshes/F3.dae",
                "knuckle"  => "meshes/knuckle.dae",
            }
        path_c ={
                "distal"   => distal_sensor_distal(sensor_type),
                "middle"   => distal_sensor_middle(sensor_type),
                "proximal" => "meshes/F3.dae",
                "knuckle"  => "meshes/knuckle.dae",
               
            }
        @mesh_c={}
        @mesh_v={}
        
        if sim_env == "gazebo"
            #model://+default path
            path_c.each do |key,value|
              @mesh_c.merge!({key => "model://shadow_hand_vijay/"+value})
            end
            path_v.each do |key,value|
              @mesh_v.merge!({key => "model://shadow_hand_vijay/"+value})
            end

        elsif sim_env == "pybullet"
            #deafult path
            @mesh_v = path_v
            @mesh_c = path_c
        else
            puts "Fingers:   well life sucks!not everything is supported.A'nt life a bitch? "
        end


    end
end


class Thumb
    attr_accessor :thbase,:thproximal,:thhub,:thmiddle,:thdistal,:palm,:mesh_c,:mesh_v,:model
    #constants
    M_PI = 3.1415926535897931
    def initialize(sim_env,load_setting,palm,sensor_type)
        @palm = palm
        @thbase =  {  # object pose
                        'link_name'=>"thbase",
                        'pose'=> [0.034,-0.0185,0.276,
                        0,0.785398,0],
                        # inetia 
                        'pose_i'=> [0,0,0,
                                    0,0,0],                   
                        'mass'=> 0.010,
                        'ixx'=> 0.00000016 ,'ixy'=> 0     ,'ixz'=> 0 ,
                        'iyy'=> 0.00000016 ,'iyz'=> 0     ,'izz'=>0.00000016 ,
                        # joint 
                        "name"       =>"thbase",
                        "p_name"       =>"palm",
                        "j_name"     =>"THJ5",
                        "j_type"     =>"revolute",
                        "j_axis"     => [0,0,1],
                        "j_limit"    => [-60/180*M_PI,60/180*M_PI], #[lower,upper]
                        "j_effort"   => 0.2,
                        "j_velocity" => 4.0,
                        "j_damping"  => 0.2,
                        "j_friction" => 0.0,
                        #visual
                        "ambient"=>  [0,0,0,1],
                        "diffuse"=>  [0,0,0,1],
                        "specular"=> [0,0,0,1],
                        "emissive"=> [0,0,0,0]
                    

            } 
        @thproximal = {  # object pose
                        'link_name'=>"thproximal",
                        'pose'=> [ 0.0+thbase['pose'][0],
                                   0.0+thbase['pose'][1],
                                   0.0+thbase['pose'][2],
                                   0,0.785398,0.0],
                        # inetia 
                        'pose_i'=> [0,0,0.019,
                                    0,0,0],                   
                        'mass'=> 0.040,
                        'ixx'=> 0.0000136 ,'ixy'=> 0     ,'ixz'=> 0 ,
                        'iyy'=> 0.0000136 ,'iyz'=> 0     ,'izz'=>0.00000313,
                        # joint 
                        "name"       =>"thproximal",
                        "p_name"     =>"thbase",
                        "j_name"     =>"THJ4",
                        "j_type"     =>"revolute",
                        "j_axis"     => [1,0,0],
                        "j_limit"    => [0,70/180*M_PI], #[lower,upper]
                        "j_effort"   => 3.0,
                        "j_velocity" => 4.0,
                        "j_damping"  => 0.2,
                        "j_friction" => 0.0,
                        #visual
                        "ambient"=>  [0,0,0,1],
                        "diffuse"=>  [0,0,0,1],
                        "specular"=> [0,0,0,1],
                        "emissive"=> [0,0,0,0]
                } 
        @thhub = {  # object pose
                    'link_name' =>"thhub",
                     'pose'=>[ 0.02687+thproximal['pose'][0],
                               0.0    +thproximal['pose'][1],
                               0.02687+thproximal['pose'][2],
                               0,0.785398,0.0],
                     # inetia 
                     'pose_i'=> [0,0,0,
                                 0,0,0],                   
                     'mass'=> 0.005,
                     'ixx'=> 0.000001 ,'ixy'=> 0     ,'ixz'=> 0 ,
                     'iyy'=> 0.000001 ,'iyz'=> 0     ,'izz'=>0.0000003 ,
                     # joint 
                     "name"       =>"thhub",
                     "p_name"     =>"thproximal",
                     "j_name"     =>"THJ3",
                     "j_type"     =>"fixed",
                     "j_axis"     => [1,0,0]
            }
        @thmiddle = {  # object pose
                  'link_name' =>"thmiddle",
                  'pose'=> [0.0+thhub['pose'][0],
                            0.0+thhub['pose'][1],
                            0.0+thhub['pose'][2],
                            0,0.785398,0.0],
                  # inetia 
                  'pose_i'=> [0,0,016,
                              0,0,0],                   
                  'mass'=> 0.020,
                  'ixx'=> 0.0000051 ,'ixy'=> 0     ,'ixz'=> 0 ,
                  'iyy'=> 0.0000051 ,'iyz'=> 0     ,'izz'=>0.00000121 ,
                  # joint 
                  "name"       =>"thmiddle",
                  "p_name"     =>"thhub",
                  "j_name"     =>"THJ2",
                  "j_type"     =>"revolute",
                  "j_axis"     => [0,-1,0],
                  "j_limit"    => [-40/180*M_PI,40/180*M_PI], #[lower,upper]
                  "j_effort"   => 2.0,
                  "j_velocity" => 2.0,
                  "j_damping"  => 0.1,
                  "j_friction" => 0.0,
                  #visual
                  "ambient"=>  [0,0,0,1],
                  "diffuse"=>  [0,0,0,1],
                  "specular"=> [0,0,0,1],
                  "emissive"=> [0,0,0,0]
            } 
        @thdistal = {  # object pose
               'link_name' =>"thdistal",
               'pose'=> [ 0.022628+thmiddle['pose'][0],
                          0.0+thmiddle['pose'][1],
                          0.022627+thmiddle['pose'][2],
                         -0.785398,0.0,-1.5708],
               # inetia 
               'pose_i'=> [0,0,01375,
                           0,0,0],                   
               'mass'=> 0.016,
               'ixx'=> 0.0000021 ,'ixy'=> 0     ,'ixz'=> 0 ,
               'iyy'=> 0.0000022 ,'iyz'=> 0     ,'izz'=>0.000001 ,
                # joint 
                "name"       =>"thdistal",
                "p_name"     =>"thmiddle",
                "j_name"     =>"THJ1",
                "j_type"     =>"revolute",
                "j_axis"     => [1,0,0],
                "j_limit"    => [0,90.0*M_PI/180], #[lower,upper]
                "j_effort"   => 1.0,
                "j_velocity" => 4.0,
                "j_damping"  => 0.2,
                "j_friction" => 0.0,
                #visual
                "ambient"=>  [0,0,0,1],
                "diffuse"=>  [0,0,0,1],
                "specular"=> [0,0,0,1],
                "emissive"=> [0,0,0,0]
        }
        @model={"TH"=>{"Links"=>{}}}
        def generate_model_forReset()
            link_list = [@thbase,@thproximal,@thhub,@thmiddle,@thdistal]
            i=0
            loop do
          
                @model["TH"]["Links"][link_list[i]["link_name"]]={ 
        
                    "mass"      =>link_list[i]["mass"],
                    "color"     =>{
                                        'ambient'  => link_list[i]["ambient"] ,                                             
                                        'specular' => link_list[i]["specular"],
                                        'emissive' => link_list[i]["emissive"],
                                        'diffuse' => link_list[i]["diffuse"] 
                    },
                    "joint"     =>{
                                        "damping"     =>link_list[i]["j_damping"],
                                        "friction"    =>link_list[i]["j_friction"]
                    }
                    }
                if i ==link_list.length-1
                 break
                end
                i +=1
           end
         
        end
        generate_model_forReset()
        def distal_sensor_distal(sensor_type)
            if sensor_type =="default"
                return  "meshes/TH1_z.dae"  
            elsif sensor_type =="eli"
                return  "meshes/thdistal_ellipsoid_visual.dae" 
            elsif sensor_type =="bio"
                return  "meshes/biotac_decimated.dae"
            end
        end
                
        def distal_sensor_middle(sensor_type)
            if sensor_type =="default"
                return  "meshes/TH2_z.dae"   
            elsif sensor_type =="eli"
                return  "meshes/TH2_z.dae"
            elsif sensor_type =="bio"
                return  "meshes/biotac_thumb_adapter.dae"
            end
        end
        def loadYamlAndChange()
            file = YAML.load_file("config.yml")
            #print file 
         
                
            #print finger
            links = file["TH"]["Links"]
            links.each do |linkName,link|

                color = link["color"]
                mass  = link["mass"] 
                joint = link["joint"]

                if linkName == "thbase"
                    #visual
                    @thbase["ambient"] = color["ambient"]
                    @thbase["diffuse"] = color["diffuse"]
                    @thbase["specular"] = color["specular"]
                    @thbase["emissive"] = color["emissive"]
                    #physical prop
                    @thbase["mass"] = mass
                    @thbase["j_damping"] = joint["damping"]
                    @thbase["j_friction"] = joint["friction"]
                elsif linkName== "thproximal"
                    #visual
                    @thproximal["ambient"] = color["ambient"]
                    @thproximal["diffuse"] = color["diffuse"]
                    @thproximal["specular"] = color["specular"]
                    @thproximal["emissive"] = color["emissive"]
                    #physical prop
                    @thproximal["mass"] = mass
                    @thproximal["j_damping"] = joint["damping"]
                    @thproximal["j_friction"] = joint["friction"]
                elsif linkName== "thhub"
                    #physical prop
                    @thhub["mass"] = mass
                    @thhub["j_damping"] = joint["damping"]
                    @thhub["j_friction"] = joint["friction"]
                elsif linkName== "thmiddle"
                    #visual
                    @thmiddle["ambient"] = color["ambient"]
                    @thmiddle["diffuse"] = color["diffuse"]
                    @thmiddle["specular"] = color["specular"]
                    @thmiddle["emissive"] = color["emissive"]
                    #physical prop
                    @thmiddle["mass"] = mass
                    @thmiddle["j_damping"] = joint["damping"]
                    @thmiddle["j_friction"] = joint["friction"]
                elsif linkName== "thdistal"
                    #visual
                    @thdistal["ambient"] = color["ambient"]
                    @thdistal["diffuse"] = color["diffuse"]
                    @thdistal["specular"] = color["specular"]
                    @thdistal["emissive"] = color["emissive"]
                    #physical prop
                    @thdistal["mass"] = mass
                    @thdistal["j_damping"] = joint["damping"]
                    @thdistal["j_friction"] = joint["friction"]
                end
            end
              
            
        end
        if load_setting == true
            loadYamlAndChange()
        end
        path_v = {
                'thproximal' => "meshes/TH3_z.dae",
                'thmiddle'   => distal_sensor_middle(sensor_type),
                'thdistal'   => distal_sensor_distal(sensor_type)
            }
        path_c = {
                'thproximal' => "meshes/TH3_z.dae",
                'thmiddle'   => distal_sensor_middle(sensor_type),
                'thdistal'   => distal_sensor_distal(sensor_type)
            }
        @mesh_c ={}
        @mesh_v ={} 
        

        if sim_env == "gazebo"
          #model://+default path
          path_c.each do |key,value|
            @mesh_c.merge!({key => "model://shadow_hand_vijay/"+value})
          end
          path_v.each do |key,value|
            @mesh_v.merge!({key => "model://shadow_hand_vijay/"+value})
          end   
        elsif sim_env == "pybullet"
          #deafult path
          @mesh_v = path_v
          @mesh_c = path_c
        else
          puts "Thumb:  well life sucks!not everything is supported.A'nt life a bitch? "
        end


            
    end
        
   
end

class LFinger
    #constants
    M_PI = 3.1415926535897931
    attr_accessor :metacarpal,:knuckle,:proximal,:middle,:distal,:palm,:mesh_c,:mesh_v,:fingername,:model
    def initialize(sim_env,load_setting,palm,biase,fingername,sensor_type)
        @fingername = fingername
        @palm = palm
        @metacarpal =  {  # object pose
                        "link_name" =>"metacarpal"+fingername,
                        'pose'=> [-0.031,-0.01,0.26771,
                        0,0,0],
                        # inetia 
                        'pose_i'=> [0,0,0.04,
                                    0,0,0],                   
                        'mass'=> 0.030,
                        'ixx'=> 0.0000145 ,'ixy'=> 0     ,'ixz'=> 0 ,
                        'iyy'=> 0.00001638 ,'iyz'=> 0     ,'izz'=>0.000004272,
                        # joint 
                        "name"       =>"metacarpal",
                        "p_name"       =>"palm",
                        "j_name"     =>"LFJ5",
                        "j_type"     =>"revolute",
                        "j_axis"     => [0,-1,0],
                        "j_limit"    => [-20/180*M_PI,20/180*M_PI], #[lower,upper]
                        "j_effort"   => 2.0,
                        "j_velocity" => 2.0,
                        "j_damping"  => 0.1,
                        "j_friction" => 0.0,
                        #color
                        "ambient"=>  [0,0,0,1],
                        "diffuse"=>  [0,0,0,1],
                        "specular"=> [0,0,0,1],
                        "emissive"=> [0,0,0,0]
            } 
        @knuckle =  {  # object pose
                        "link_name" =>"knuckle"+fingername,
                        'pose'=> [0.0+metacarpal['pose'][0],
                                   0+metacarpal['pose'][1],
                                   0.06579+metacarpal['pose'][2],
                                   0,0,0],
                        # inetia 
                        'pose_i'=> [0,0,0,
                                    0,0,0],                   
                        'mass'=> 0.008,
                        'ixx'=> 0.00000026 ,'ixy'=> 0     ,'ixz'=> 0 ,
                        'iyy'=> 0.00000026 ,'iyz'=> 0     ,'izz'=>0.00000032,
                        # joint 
                        "name"       =>"knuckle",
                        "p_name"       =>"metacarpal"+fingername,
                        "j_name"     =>"J4_"+fingername,
                        "j_type"     =>"revolute",
                        "j_axis"     => [0,-1,0],
                        "j_limit"    => [-20/180*M_PI,20/180*M_PI], #[lower,upper]
                        "j_effort"   => 2.0,
                        "j_velocity" => 2.0,
                        "j_damping"  => 0.1,
                        "j_friction" => 0.0,
                        #color
                        "ambient"=>  [0,0,0,1],
                        "diffuse"=>  [0,0,0,1],
                        "specular"=> [0,0,0,1],
                        "emissive"=> [0,0,0,0] 
            } 
        @proximal = {  # object pose
                    "link_name" =>"proximal"+fingername,
                    'pose'=> [0+knuckle['pose'][0],
                            0+knuckle['pose'][1],
                            0+knuckle['pose'][2],
                            0,0,0],
                    # inetia 
                    'pose_i'=> [0,0,0.0225,
                                    0,0,0],                   
                    'mass'=> 0.030,
                    'ixx'=> 0.0000098 ,'ixy'=> 0     ,'ixz'=> 0 ,
                    'iyy'=> 0.00001 ,'iyz'=> 0     ,'izz'=>0.0000018 ,
                     # joint 
                     "name"       =>"proximal",
                     "p_name"     =>"knuckle"+fingername,
                     "j_name"     =>"J3_"+fingername,
                     "j_type"     =>"revolute",
                     "j_axis"     => [1,0,0],
                     "j_limit"    => [0,M_PI/2], #[lower,upper]
                     "j_effort"   => 2.0,
                     "j_velocity" => 2.0,
                     "j_damping"  => 0.1 ,
                     "j_friction" => 0.0,                      
                    #color
                     "ambient"=>  [0,0,0,1],
                     "diffuse"=>  [0,0,0,1],
                     "specular"=> [0,0,0,1],
                     "emissive"=> [0,0,0,0] 
            } 
        @middle =   {  # object pose
                    "link_name" =>"middle"+fingername,
                    'pose'=> [0+proximal['pose'][0],
                            0+proximal['pose'][1],
                            0.045+proximal['pose'][2],
                            0,0,0],
                    # inetia 
                    'pose_i'=> [0,0,0.0125,
                                    0,0,0],                   
                    'mass'=> 0.017,
                    'ixx'=>0.0000026 ,'ixy'=> 0     ,'ixz'=> 0 ,
                    'iyy'=>0.0000027 ,'iyz'=> 0     ,'izz'=>0.00000087,
                    # joint 
                    "name"       =>"middle",
                    "p_name"     =>"proximal"+fingername,
                    "j_name"     =>"J2_"+fingername,
                    "j_type"     =>"revolute",
                    "j_axis"     => [1,0,0],
                    "j_limit"    => [0,M_PI/2], #[lower,upper]
                    "j_effort"   => 2.0,
                    "j_velocity" => 2.0,
                    "j_damping"  => 0.1 ,
                    "j_friction" => 0.0,
                    #color
                    "ambient"=>  [0,0,0,1],
                    "diffuse"=>  [0,0,0,1],
                    "specular"=> [0,0,0,1],
                    "emissive"=> [0,0,0,0] 
                               
            }
        @distal =   {  # object pose
                "link_name" =>"distal"+fingername,
                'pose'=> [0+middle['pose'][0],
                          0+middle['pose'][1],
                          0.025+middle['pose'][2],
                          0,0,0],
                # inetia 
                'pose_i'=> [0,0,0.012,
                                0,0,0],                   
                'mass'=> 0.012,
                'ixx'=>0.00000094 ,'ixy'=> 0     ,'ixz'=> 0 ,
                'iyy'=>0.0000011 ,'iyz'=> 0     ,'izz'=>0.00000053,
                # joint 
                "name"       =>"distal",
                "p_name"     =>"middle"+fingername,
                "j_name"     =>"J1_"+fingername,
                "j_type"     =>"revolute",
                "j_axis"     => [1,0,0],
                "j_limit"    => [0,M_PI/2], #[lower,upper]
                "j_effort"   => 2.0,
                "j_velocity" => 2.0,
                "j_damping"  => 0.1,
                "j_friction" => 0.0,
                #color
                "ambient"=>  [0,0,0,1],
                "diffuse"=>  [0,0,0,1],
                "specular"=> [0,0,0,1],
                "emissive"=> [0,0,0,0] 
            
        }
        @model={"LF"=>{"Links"=>{}}}
        def generate_model_forReset()
            link_list = [@metacarpal,@knuckle,@proximal,@middle,@distal]
            i=0
            loop do
          
                @model["LF"]["Links"][link_list[i]["link_name"]]={ 
        
                    "mass"      =>link_list[i]["mass"],

                    "color"     =>{
                        'ambient'  => link_list[i]["ambient"] ,                                             
                        'specular' => link_list[i]["specular"],
                        'emissive' => link_list[i]["emissive"],
                        'diffuse' => link_list[i]["diffuse"] 
                    },
                    "joint"     =>{
                        "damping"     =>link_list[i]["j_damping"],
                        "friction"    =>link_list[i]["j_friction"]
                    }
                }
                if i ==link_list.length-1
                 break
                end
                i +=1
           end
        end
        generate_model_forReset()
        def distal_sensor_distal(sensor_type)
            if sensor_type =="default"
                return  "meshes/F1.dae"   
            elsif sensor_type =="eli"
                return  "meshes/distal_ellipsoid_visual.dae" 
            elsif sensor_type =="bio"
                return  "meshes/biotac_decimated.dae"
            end
        end
        #biotac_finger_adapter.dae
        def distal_sensor_middle(sensor_type)
            if sensor_type =="default"
                return  "meshes/F2.dae"   
            elsif sensor_type =="eli"
                return  "meshes/F2.dae"
            elsif sensor_type =="bio"
                return  "meshes/biotac_finger_adapter.dae"
            end
        end
        def loadYamlAndChange()
            file = YAML.load_file("config.yml")
            #print file 
         
                
            #print finger
            links = file["LF"]["Links"]
            links.each do |linkName,link|

                color = link["color"]
                mass  = link["mass"] 
                joint = link["joint"]

                if linkName == "metacarpal"
                    #visual
                    @metacarpal["ambient"] = color["ambient"]
                    @metacarpal["diffuse"] = color["diffuse"]
                    @metacarpal["specular"] = color["specular"]
                    @metacarpal["emissive"] = color["emissive"]
                    #physical prop
                    @metacarpal["mass"] = mass
                    @metacarpal["j_damping"] = joint["damping"]
                    @metacarpal["j_friction"] = joint["friction"]
                elsif linkName == "knuckle"
                    #visual
                    @knuckle["ambient"] = color["ambient"]
                    @knuckle["diffuse"] = color["diffuse"]
                    @knuckle["specular"] = color["specular"]
                    @knuckle["emissive"] = color["emissive"]
                    #physical prop
                    @knuckle["mass"] = mass
                    @knuckle["j_damping"] = joint["damping"]
                    @knuckle["j_friction"] = joint["friction"]
                elsif linkName== "proximal"
                    #visual
                    @proximal["ambient"] = color["ambient"]
                    @proximal["diffuse"] = color["diffuse"]
                    @proximal["specular"] = color["specular"]
                    @proximal["emissive"] = color["emissive"]
                    #physical prop
                    @proximal["mass"] = mass
                    @proximal["j_damping"] = joint["damping"]
                    @proximal["j_friction"] = joint["friction"]
                elsif linkName== "middle"
                    #visual
                    @middle["ambient"] = color["ambient"]
                    @middle["diffuse"] = color["diffuse"]
                    @middle["specular"] = color["specular"]
                    @middle["emissive"] = color["emissive"]
                    #physical prop
                    @middle["mass"] = mass
                    @middle["j_damping"] = joint["damping"]
                    @middle["j_friction"] = joint["friction"]
                elsif linkName== "distal"
                    #visual
                    @distal["ambient"] = color["ambient"]
                    @distal["diffuse"] = color["diffuse"]
                    @distal["specular"] = color["specular"]
                    @distal["emissive"] = color["emissive"]
                    #physical prop
                    @distal["mass"] = mass
                    @distal["j_damping"] = joint["damping"]
                    @distal["j_friction"] = joint["friction"]
                end
            end
              
            
        end
        if load_setting == true
            loadYamlAndChange()
        end
        path_v ={
                "distal"   => distal_sensor_distal(sensor_type),
                "middle"   => distal_sensor_middle(sensor_type),
                "proximal" => "meshes/F3.dae",
                "knuckle"  => "meshes/knuckle.dae",
                "lfmetacarpal"  => "meshes/lfmetacarpal.dae"
            }
        path_c ={
                "distal"   => distal_sensor_distal(sensor_type),
                "middle"   => distal_sensor_middle(sensor_type),
                "proximal" => "meshes/F3.dae",
                "knuckle"  => "meshes/knuckle.dae",
                "lfmetacarpal"  => "meshes/lfmetacarpal.dae"
            }
 
 
        @mesh_c={}
        @mesh_v={}

        if sim_env == "gazebo"
          #model://+default path
          path_c.each do |key,value|
            @mesh_c.merge!({key => "model://shadow_hand_vijay/"+value})
          end
          path_v.each do |key,value|
            @mesh_v.merge!({key => "model://shadow_hand_vijay/"+value})
          end

        elsif sim_env == "pybullet"
          #deafult path
          @mesh_v = path_v
          @mesh_c = path_c
        else
          puts "little finger:  well life sucks!not everything is supported.A'nt life a bitch? "
        end
        

    
    end

end


arg_bool = false
if  ARGV.length >0
    arg = ARGV[0]
    if arg =="true" 
        arg_bool = true
    elsif arg =="false" 
        arg_bool = false
    end
 end
 

 load_setting = arg_bool || false



sim_env = "pybullet"
base =Base.new(sim_env,load_setting)   

biase = [0,0,0]
ff = Finger.new(sim_env,load_setting,base.palm,biase,"FF","bio")
# print ff.model
# puts ff.middle["diffuse"]
# puts ff.knuckle["mass"]
# puts ff.knuckle["j_damping"]
# puts ff.knuckle["j_friction"]



mf = Finger.new(sim_env,load_setting,base.palm,biase,"MF","bio")
#print mf.model
#puts mf.distal["specular"]
rf = Finger.new(sim_env,load_setting,base.palm,biase,"RF","bio")
thumb = Thumb.new(sim_env,load_setting,base.palm,"eli")
# print thumb.model
#puts thumb.thbase["mass"]
#puts thumb.thbase["ambient"]
#puts thumb.thbase["j_damping"]
#puts thumb.thbase["j_friction"]

#puts thumb.thhub["mass"]
#puts thumb.thhub["j_damping"]
#puts thumb.thhub["j_friction"]

lf = LFinger.new(sim_env,load_setting,base.palm,biase,"LF","eli")
#print lf.model
#print lf.metacarpal["diffuse"]



#reset the model
def rest_modelConfigYML(base,ff,mf,rf,lf,thumb)
    robot_model = {
                    "BASE"=>base.model["BASE"],
                    "FF"=>ff.model["FF"],
                    "MF"=>mf.model["MF"],
                    "RF"=>rf.model["RF"],
                    "TH"=>thumb.model["TH"],
                    "LF"=>lf.model["LF"] 
    }
    File.open("config.yml", "w") do |out|
        YAML.dump( robot_model, out )
      
    end
end
rest_modelConfigYML(base,ff,mf,rf,lf,thumb)