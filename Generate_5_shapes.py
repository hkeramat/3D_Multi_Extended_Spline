# Generic imports
import os
import glob
import math
import time
import PIL
import matplotlib
import numpy             as np
import matplotlib.pyplot as plt

# Custom imports
from shapes_utils  import *

from shapes_utils_1  import *
from shapes_utils_2  import *
from shapes_utils_3  import *
from shapes_utils_4  import *

from meshes_utils  import *
from fenics_solver import *
from tensorforce import Environment, Runner
# Define environment class for rl
class env(Environment):

    def states(self):
        return dict(
            type='float',
            shape=( 3*env.shape.n_control_pts))

    
    def actions(self):
        return dict(
            type='float',
            shape=(self.nb_pts_to_move*3),
            min_value=-1.0,
            max_value= 1.0)


    

    # Static variable
    episode_nb =-1
    control_nb = 0

    def num_actors(self):
            return 5  # Indicates that environment has multiple actors
    
    

    

    # Initialize empty shape
    shape = Shape()
    shape_1 = Shape_1()
    shape_2 = Shape_2()
    shape_3 = Shape_3()
    shape_4 = Shape_4()

    
    def __init__(self,
                 nb_pts_to_move, pts_to_move,
                 nb_ctrls_per_episode, nb_episodes,
                 max_deformation,
                 restart_from_cylinder,
                 replace_shape,
                 comp_dir,
                 restore_model,
                 saving_model_period,
                 final_time, cfl, reynolds,
                 output,
                 shape_h, domain_h,
                 cell_limit,
                 reset_dir,
                 xmin, xmax, ymin, ymax):

        self.nb_pts_to_move          = nb_pts_to_move
        self.pts_to_move             = pts_to_move
        self.nb_ctrls_per_episode    = nb_ctrls_per_episode
        self.nb_episodes             = nb_episodes
        self.max_deformation         = max_deformation
        self.restart_from_cylinder   = restart_from_cylinder
        self.replace_shape           = replace_shape
        self.comp_dir                = comp_dir
        self.restore_model           = restore_model
        self.final_time              = final_time
        self.cfl                     = cfl
        self.reynolds                = reynolds
        self.output                  = output
        self.shape_h                 = shape_h
        self.domain_h                = domain_h
        self.cell_limit              = cell_limit
        self.reset_dir               = reset_dir
        self.xmin                    = xmin
        self.xmax                    = xmax
        self.ymin                    = ymin
        self.ymax                    = ymax
        

        

        # Saving model periodically
        env.saving_model_period = saving_model_period
           

        # Check that reset dir exists
        if (not os.path.exists('./'+self.reset_dir)):
            print('Error : I could not find the reset folder')
            exit()

        # Initialize shape by reading it from reset folder
        # Shape reset is automatic when reading from csv
        env.shape.read_csv(self.reset_dir+'/shape_0.csv')
        env.shape.generate(centering=False)

        env.shape_1.read_csv(self.reset_dir+'/shape_0.csv')
        env.shape_1.generate(centering=False)

        env.shape_2.read_csv(self.reset_dir+'/shape_0.csv')
        env.shape_2.generate(centering=False)
        
        env.shape_3.read_csv(self.reset_dir+'/shape_0.csv')
        env.shape_3.generate(centering=False)

        env.shape_4.read_csv(self.reset_dir+'/shape_0.csv')
        env.shape_4.generate(centering=False)

        
        # Initialize arrays
        self.pressure_drop       = np.array([])
        self.heat       = np.array([])
        self.reward     = np.array([])
        self.avg_pressure_drop   = np.array([])
        self.avg_heat   = np.array([])
        self.avg_reward = np.array([])
        self.penal      = np.array([])

        # If restore model, get last increment
        if (self.restore_model):
            file_lst        = glob.glob(self.comp_dir+'/save/png/*.png')
            last_file       = max(file_lst, key=os.path.getctime)
            tmp             = last_file.split('_')[-1]
            env.shape.index = int(tmp.split('.')[0])
            print('Restarting from shape index '+str(env.shape.index))

        # Remove save folder
        if (not self.restore_model):
            if (os.path.exists(self.comp_dir+'/save')):
                os.system('rm -r '+self.comp_dir+'/save')

            # Make sure the save repo exists and is properly formated
            if (not os.path.exists(self.comp_dir+'/save')):
                os.system('mkdir '+self.comp_dir+'/save')
            if (not os.path.exists(self.comp_dir+'/save/png')):
                os.system('mkdir '+self.comp_dir+'/save/png')
            if (not os.path.exists(self.comp_dir+'/save/rejected')):
                os.system('mkdir '+self.comp_dir+'/save/rejected')
            if (not os.path.exists(self.comp_dir+'/save/xml')):
                os.system('mkdir '+self.comp_dir+'/save/xml')
            if (not os.path.exists(self.comp_dir+'/save/csv')):
                os.system('mkdir '+self.comp_dir+'/save/csv')
            if (not os.path.exists(self.comp_dir+'/save/sol')):
                os.system('mkdir '+self.comp_dir+'/save/sol')

            # Copy initial files in save repo if restart from cylinder
            if (self.restart_from_cylinder):
                os.system('cp '+self.reset_dir+'/shape_0.png '+self.comp_dir+'/save/png/.')
                os.system('cp '+self.reset_dir+'/shape_0.xml '+self.comp_dir+'/save/xml/.')
                os.system('cp '+self.reset_dir+'/shape_0.csv '+self.comp_dir+'/save/csv/.')

    def reset(self):

        # Always for multi-actor environments: initialize parallel indices
        self._parallel_indices = np.arange(self.num_actors())

        # Single shared environment logic, plus per-actor perspective
        self.fifth_actor = True
        # Console output
        env.episode_nb += 1
        print('****** Starting episode '+str(env.episode_nb))
        if (env.episode_nb%100 == 0): time.sleep(10)

        # Reset control number
        env.control_nb  = 0

        # Reset from cylinder if asked
        if (self.restart_from_cylinder):
            env.shape.read_csv(self.reset_dir+'/shape_0.csv', keep_numbering=True)
            env.shape.generate(centering=False)

        # Fill next state
        next_state_0 = self.fill_next_state(True, 0)
        next_state_1 = self.fill_next_state_1(True, 0)
        next_state_2 = self.fill_next_state_2(True, 0)
        next_state_3 = self.fill_next_state_3(True, 0)
        next_state_4 = self.fill_next_state_4(True, 0)

        
        next_state = np.stack([next_state_0, next_state_1,next_state_2, next_state_3,next_state_4],axis=0)

        return self._parallel_indices.copy(), next_state

    def execute(self, actions):
        # Console output
        print('***    Starting control '+str(env.control_nb))
        # Single shared environment logic, plus per-actor perspective
        if self.fifth_actor:
            
            terminal = np.stack([False, False, False, False,not self.fifth_actor], axis=0)
            # Convert actions to numpy array
            
            deformation = np.array(actions[0]).reshape((int(len(actions[0])/3), 3))
            print('deformation {}'.format(deformation))
            deformation_1 = np.array(actions[1]).reshape((int(len(actions[1])/3), 3))
            print('deormation 1 {}'.format(deformation_1))
            deformation_2 = np.array(actions[2]).reshape((int(len(actions[2])/3), 3))
            print('deormation 2 {}'.format(deformation_2))
            deformation_3 = np.array(actions[3]).reshape((int(len(actions[3])/3), 3))
            print('deormation 3 {}'.format(deformation_3))
            deformation_4 = np.array(actions[4]).reshape((int(len(actions[4])/3), 3))
            print('deormation 4 {}'.format(deformation_4))


            for i in range(self.nb_pts_to_move):
                pt     = self.pts_to_move[i]
                
                radius = max(abs(deformation[i,0]),0.2)*self.max_deformation
                
                dangle = (360.0/float(env.shape.n_control_pts))
                
                angle  = dangle*float(pt)+deformation[i,1]*dangle/2.0
                
                x      = radius*math.cos(math.radians(angle))
                y      = radius*math.sin(math.radians(angle))
                edg    = 0.5+0.5*abs(deformation[i,2])
                
                deformation[i,0] = x
                deformation[i,1] = y
                deformation[i,2] = edg
            
            for i in range(self.nb_pts_to_move):
                pt     = self.pts_to_move[i]
                
                radius = max(abs(deformation_1[i,0]),0.2)*self.max_deformation
                
                dangle = (360.0/float(env.shape_1.n_control_pts_1))
                
                angle  = dangle*float(pt)+deformation_1[i,1]*dangle/2.0
                
                x      = radius*math.cos(math.radians(angle))
                y      = radius*math.sin(math.radians(angle))
                edg    = 0.5+0.5*abs(deformation_1[i,2])
                
                deformation_1[i,0] = x
                deformation_1[i,1] = y
                deformation_1[i,2] = edg

            for i in range(self.nb_pts_to_move):
                pt     = self.pts_to_move[i]
                
                radius = max(abs(deformation_2[i,0]),0.2)*self.max_deformation
                
                dangle = (360.0/float(env.shape_2.n_control_pts_2))
                
                angle  = dangle*float(pt)+deformation_2[i,1]*dangle/2.0
                
                x      = radius*math.cos(math.radians(angle))
                y      = radius*math.sin(math.radians(angle))
                edg    = 0.5+0.5*abs(deformation_2[i,2])
                
                deformation_2[i,0] = x
                deformation_2[i,1] = y
                deformation_2[i,2] = edg

            for i in range(self.nb_pts_to_move):
                pt     = self.pts_to_move[i]
                
                radius = max(abs(deformation_3[i,0]),0.2)*self.max_deformation
                
                dangle = (360.0/float(env.shape_3.n_control_pts_3))
                
                angle  = dangle*float(pt)+deformation_3[i,1]*dangle/2.0
                
                x      = radius*math.cos(math.radians(angle))
                y      = radius*math.sin(math.radians(angle))
                edg    = 0.5+0.5*abs(deformation_3[i,2])
                
                deformation_3[i,0] = x
                deformation_3[i,1] = y
                deformation_3[i,2] = edg
            
            for i in range(self.nb_pts_to_move):
                pt     = self.pts_to_move[i]
                
                radius = max(abs(deformation_4[i,0]),0.2)*self.max_deformation
                
                dangle = (360.0/float(env.shape_4.n_control_pts_4))
                
                angle  = dangle*float(pt)+deformation_4[i,1]*dangle/2.0
                
                x      = radius*math.cos(math.radians(angle))
                y      = radius*math.sin(math.radians(angle))
                edg    = 0.5+0.5*abs(deformation_4[i,2])
                
                deformation_4[i,0] = x
                deformation_4[i,1] = y
                deformation_4[i,2] = edg
            # Modify shape
            env.shape.modify_shape_from_field(deformation,
                                            replace=self.replace_shape,
                                            pts_list=self.pts_to_move)
            env.shape_1.modify_shape_from_field(deformation_1,
                                            replace=self.replace_shape,
                                            pts_list=self.pts_to_move)
            env.shape_2.modify_shape_from_field(deformation_2,
                                            replace=self.replace_shape,
                                            pts_list=self.pts_to_move)
            env.shape_3.modify_shape_from_field(deformation_3,
                                            replace=self.replace_shape,
                                            pts_list=self.pts_to_move)
            env.shape_4.modify_shape_from_field(deformation_4,
                                            replace=self.replace_shape,
                                            pts_list=self.pts_to_move)
            if (    self.replace_shape):           centering = True
            if (not self.replace_shape):           centering = False

            env.shape.generate(centering=False)
            env.shape.write_csv()
            env.shape_1.generate(centering=False)
            env.shape_1.write_csv()
            env.shape_2.generate(centering=False)
            env.shape_2.write_csv()
            env.shape_3.generate(centering=False)
            env.shape_3.write_csv()
            env.shape_4.generate(centering=False)
            env.shape_4.write_csv()
        elif self.fourth_actor:
            terminal = np.stack([False, False, False,not self.fourth_actor], axis=0)
            # Convert actions to numpy array
            
            deformation = np.array(actions[0]).reshape((int(len(actions[0])/3), 3))
            print('deformation {}'.format(deformation))
            deformation_1 = np.array(actions[1]).reshape((int(len(actions[1])/3), 3))
            print('deormation 1 {}'.format(deformation_1))
            deformation_2 = np.array(actions[2]).reshape((int(len(actions[2])/3), 3))
            print('deormation 2 {}'.format(deformation_2))
            deformation_3 = np.array(actions[3]).reshape((int(len(actions[3])/3), 3))
            print('deormation 3 {}'.format(deformation_3))


            for i in range(self.nb_pts_to_move):
                pt     = self.pts_to_move[i]
                
                radius = max(abs(deformation[i,0]),0.2)*self.max_deformation
                
                dangle = (360.0/float(env.shape.n_control_pts))
                
                angle  = dangle*float(pt)+deformation[i,1]*dangle/2.0
                
                x      = radius*math.cos(math.radians(angle))
                y      = radius*math.sin(math.radians(angle))
                edg    = 0.5+0.5*abs(deformation[i,2])
                
                deformation[i,0] = x
                deformation[i,1] = y
                deformation[i,2] = edg
            
            for i in range(self.nb_pts_to_move):
                pt     = self.pts_to_move[i]
                
                radius = max(abs(deformation_1[i,0]),0.2)*self.max_deformation
                
                dangle = (360.0/float(env.shape_1.n_control_pts_1))
                
                angle  = dangle*float(pt)+deformation_1[i,1]*dangle/2.0
                
                x      = radius*math.cos(math.radians(angle))
                y      = radius*math.sin(math.radians(angle))
                edg    = 0.5+0.5*abs(deformation_1[i,2])
                
                deformation_1[i,0] = x
                deformation_1[i,1] = y
                deformation_1[i,2] = edg

            for i in range(self.nb_pts_to_move):
                pt     = self.pts_to_move[i]
                
                radius = max(abs(deformation_2[i,0]),0.2)*self.max_deformation
                
                dangle = (360.0/float(env.shape_2.n_control_pts_2))
                
                angle  = dangle*float(pt)+deformation_2[i,1]*dangle/2.0
                
                x      = radius*math.cos(math.radians(angle))
                y      = radius*math.sin(math.radians(angle))
                edg    = 0.5+0.5*abs(deformation_2[i,2])
                
                deformation_2[i,0] = x
                deformation_2[i,1] = y
                deformation_2[i,2] = edg

            for i in range(self.nb_pts_to_move):
                pt     = self.pts_to_move[i]
                
                radius = max(abs(deformation_3[i,0]),0.2)*self.max_deformation
                
                dangle = (360.0/float(env.shape_3.n_control_pts_3))
                
                angle  = dangle*float(pt)+deformation_3[i,1]*dangle/2.0
                
                x      = radius*math.cos(math.radians(angle))
                y      = radius*math.sin(math.radians(angle))
                edg    = 0.5+0.5*abs(deformation_3[i,2])
                
                deformation_3[i,0] = x
                deformation_3[i,1] = y
                deformation_3[i,2] = edg
            # Modify shape
            env.shape.modify_shape_from_field(deformation,
                                            replace=self.replace_shape,
                                            pts_list=self.pts_to_move)
            env.shape_1.modify_shape_from_field(deformation_1,
                                            replace=self.replace_shape,
                                            pts_list=self.pts_to_move)
            env.shape_2.modify_shape_from_field(deformation_2,
                                            replace=self.replace_shape,
                                            pts_list=self.pts_to_move)
            env.shape_3.modify_shape_from_field(deformation_3,
                                            replace=self.replace_shape,
                                            pts_list=self.pts_to_move)
            if (    self.replace_shape):           centering = True
            if (not self.replace_shape):           centering = False

            env.shape.generate(centering=False)
            env.shape.write_csv()
            env.shape_1.generate(centering=False)
            env.shape_1.write_csv()
            env.shape_2.generate(centering=False)
            env.shape_2.write_csv()
            env.shape_3.generate(centering=False)
            env.shape_3.write_csv()

        elif self.third_actor:
            terminal = np.stack([False, False, not self.third_actor], axis=0)
            # Convert actions to numpy array
            print('actions 0 {}'.format(actions[0]))
            print('actions 1 {}'.format(actions[1]))
            print('actions 2 {}'.format(actions[2]))
            deformation = np.array(actions[0]).reshape((int(len(actions[0])/3), 3))
            print('deformation {}'.format(deformation))
            deformation_1 = np.array(actions[1]).reshape((int(len(actions[1])/3), 3))
            print('deormation 1 {}'.format(deformation_1))
            deformation_2 = np.array(actions[2]).reshape((int(len(actions[2])/3), 3))
            print('deormation 2 {}'.format(deformation_2))


            for i in range(self.nb_pts_to_move):
                pt     = self.pts_to_move[i]
                
                radius = max(abs(deformation[i,0]),0.2)*self.max_deformation
                
                dangle = (360.0/float(env.shape.n_control_pts))
                
                angle  = dangle*float(pt)+deformation[i,1]*dangle/2.0
                
                x      = radius*math.cos(math.radians(angle))
                y      = radius*math.sin(math.radians(angle))
                edg    = 0.5+0.5*abs(deformation[i,2])
                
                deformation[i,0] = x
                deformation[i,1] = y
                deformation[i,2] = edg
            
            for i in range(self.nb_pts_to_move):
                pt     = self.pts_to_move[i]
                
                radius = max(abs(deformation_1[i,0]),0.2)*self.max_deformation
                
                dangle = (360.0/float(env.shape_1.n_control_pts_1))
                
                angle  = dangle*float(pt)+deformation_1[i,1]*dangle/2.0
                
                x      = radius*math.cos(math.radians(angle))
                y      = radius*math.sin(math.radians(angle))
                edg    = 0.5+0.5*abs(deformation_1[i,2])
                
                deformation_1[i,0] = x
                deformation_1[i,1] = y
                deformation_1[i,2] = edg

            for i in range(self.nb_pts_to_move):
                pt     = self.pts_to_move[i]
                
                radius = max(abs(deformation_2[i,0]),0.2)*self.max_deformation
                
                dangle = (360.0/float(env.shape_2.n_control_pts_2))
                
                angle  = dangle*float(pt)+deformation_2[i,1]*dangle/2.0
                
                x      = radius*math.cos(math.radians(angle))
                y      = radius*math.sin(math.radians(angle))
                edg    = 0.5+0.5*abs(deformation_2[i,2])
                
                deformation_2[i,0] = x
                deformation_2[i,1] = y
                deformation_2[i,2] = edg

            # Modify shape
            env.shape.modify_shape_from_field(deformation,
                                            replace=self.replace_shape,
                                            pts_list=self.pts_to_move)
            env.shape_1.modify_shape_from_field(deformation_1,
                                            replace=self.replace_shape,
                                            pts_list=self.pts_to_move)
            env.shape_2.modify_shape_from_field(deformation_2,
                                            replace=self.replace_shape,
                                            pts_list=self.pts_to_move)
            if (    self.replace_shape):           centering = True
            if (not self.replace_shape):           centering = False

            env.shape.generate(centering=False)
            env.shape.write_csv()
            env.shape_1.generate(centering=False)
            env.shape_1.write_csv()
            env.shape_2.generate(centering=False)
            env.shape_2.write_csv()

        elif self.second_actor:

            terminal = np.stack([False, not self.second_actor], axis=0)

            # Convert actions to numpy array
            print('actions 0 {}'.format(actions[0]))
            print('actions 1 {}'.format(actions[1]))
            deformation = np.array(actions[0]).reshape((int(len(actions[0])/3), 3))
            print('deformation {}'.format(deformation))
            deformation_1 = np.array(actions[1]).reshape((int(len(actions[1])/3), 3))
            print('deormation 1 {}'.format(deformation_1))


            for i in range(self.nb_pts_to_move):
                pt     = self.pts_to_move[i]
                
                radius = max(abs(deformation[i,0]),0.2)*self.max_deformation
                
                dangle = (360.0/float(env.shape.n_control_pts))
                
                angle  = dangle*float(pt)+deformation[i,1]*dangle/2.0
                
                x      = radius*math.cos(math.radians(angle))
                y      = radius*math.sin(math.radians(angle))
                edg    = 0.5+0.5*abs(deformation[i,2])
                
                deformation[i,0] = x
                deformation[i,1] = y
                deformation[i,2] = edg
            
            for i in range(self.nb_pts_to_move):
                pt     = self.pts_to_move[i]
                
                radius = max(abs(deformation_1[i,0]),0.2)*self.max_deformation
                
                dangle = (360.0/float(env.shape_1.n_control_pts_1))
                
                angle  = dangle*float(pt)+deformation_1[i,1]*dangle/2.0
                
                x      = radius*math.cos(math.radians(angle))
                y      = radius*math.sin(math.radians(angle))
                edg    = 0.5+0.5*abs(deformation_1[i,2])
                
                deformation_1[i,0] = x
                deformation_1[i,1] = y
                deformation_1[i,2] = edg

            # Modify shape
            env.shape.modify_shape_from_field(deformation,
                                            replace=self.replace_shape,
                                            pts_list=self.pts_to_move)
            env.shape_1.modify_shape_from_field(deformation_1,
                                            replace=self.replace_shape,
                                            pts_list=self.pts_to_move)
            if (    self.replace_shape):           centering = True
            if (not self.replace_shape):           centering = False

            env.shape.generate(centering=False)
            env.shape.write_csv()
            env.shape_1.generate(centering=False)
            env.shape_1.write_csv()
        else:
            terminal = np.stack([False], axis=0)
            # Convert actions to numpy array
            print(actions[0])
            deformation = np.array(actions[0]).reshape((int(len(actions[0])/3), 3))
            print(deformation)
            for i in range(self.nb_pts_to_move):
                pt     = self.pts_to_move[i]
                
                radius = max(abs(deformation[i,0]),0.2)*self.max_deformation
                
                dangle = (360.0/float(env.shape.n_control_pts))
                
                angle  = dangle*float(pt)+deformation[i,1]*dangle/2.0
                
                x      = radius*math.cos(math.radians(angle))
                
                y      = radius*math.sin(math.radians(angle))
                
                edg    = 0.5+0.5*abs(deformation[i,2])

                deformation[i,0] = x
                deformation[i,1] = y
                deformation[i,2] = edg

            # Modify shape
            env.shape.modify_shape_from_field(deformation,
                                            replace=self.replace_shape,
                                            pts_list=self.pts_to_move)
            if (    self.replace_shape):           centering = True
            if (not self.replace_shape):           centering = False
            env.shape.generate(centering=False)
        

        self.curve_pts_1 = env.shape_1.curve_pts_1
        self.curve_pts_2 = env.shape_2.curve_pts_2
        self.curve_pts_3 = env.shape_3.curve_pts_3
        self.curve_pts_4 = env.shape_4.curve_pts_4

        try:
            meshed, n_tri = env.shape.mesh(mesh_domain = True,
                                           shape_h     = self.shape_h,
                                           curve_pts_1 = self.curve_pts_1,
                                           curve_pts_2 = self.curve_pts_2,
                                           curve_pts_3 = self.curve_pts_3,
                                           curve_pts_4 = self.curve_pts_4,
                                           domain_h    = self.domain_h,
                                           xmin        = self.xmin,
                                           xmax        = self.xmax,
                                           ymin        = self.ymin,
                                           ymax        = self.ymax,
                                           mesh_format = 'xml')

            # Do not solve if mesh is too large
            if (n_tri > self.cell_limit):
                meshed = False
                os.system('cp '+env.shape.name+'_'+str(env.shape.index)+'.png '
                          +self.comp_dir+'/save/rejected/.')
        except Exception as exc:
            print(exc)
            meshed = False

        # Generate image
        env.shape.generate_image(plot_pts    = True,
                                 quad_radius = self.max_deformation,
                                 xmin        = self.xmin,
                                 xmax        = self.xmax,
                                 ymin        = self.ymin,
                                 ymax        = self.ymax)

        # Save png and csv files
        os.system('mv '+env.shape.name+'_'+str(env.shape.index)+'.png '
                  +self.comp_dir+'/save/png/.')
        os.system('mv '+env.shape.name+'_'+str(env.shape.index)+'.csv '
                  +self.comp_dir+'/save/csv/.')

        # Copy new shape files to save folder
        if (meshed):
            os.system('cp '+env.shape.name+'_'+str(env.shape.index)+'.xml '
                      +self.comp_dir+'/save/xml/.')

        # Update control number
        env.control_nb += 1

        # Compute reward with try/catch
        self.compute_reward(meshed)

        # Save quantities of interest
        self.save_qoi()

        # Fill next state
        next_state_0 = self.fill_next_state(meshed, env.shape.index)
        next_state_1 = self.fill_next_state_1(meshed, env.shape_1.index_1)
        next_state_2 = self.fill_next_state_2(meshed, env.shape_2.index_2)
        next_state_3 = self.fill_next_state_3(meshed, env.shape_3.index_3)
        next_state_4 = self.fill_next_state_4(meshed, env.shape_4.index_4)
        # Copy u, v and p solutions to repo
        if (meshed):
            os.system('mv '+str(env.shape.index)+'_u.png '+self.comp_dir+'/save/sol/.')
            os.system('mv '+str(env.shape.index)+'_v.png '+self.comp_dir+'/save/sol/.')
            os.system('mv '+str(env.shape.index)+'_p.png '+self.comp_dir+'/save/sol/.')        
        # Remove mesh file from repo
        if (meshed):
            os.system('rm '+env.shape.name+'_'+str(env.shape.index)+'.xml')

        # Return
        
        print("good epoch; reward: {}".format(self.reward[-1]))
        states = np.stack([next_state_0, next_state_1, next_state_2, next_state_3, next_state_4],axis=0)
        print('states {}'.format(states))
        reward = [self.reward[-1],self.reward[-1],self.reward[-1], self.reward[-1], self.reward[-1]]
        # Always for multi-actor environments: update parallel indices, and return per-actor values
        self._parallel_indices = self._parallel_indices[~terminal]
        return self._parallel_indices.copy(), states, terminal, reward



    def compute_reward(self, meshed):
        # If meshing was successful, reward is computed normally
        if (meshed):
            try:
                # Compute pressure_drop and heat
                name = self.comp_dir+'/'+env.shape.name+'_'+str(env.shape.index)+'.xml'
                pressure_drop, heat, solved = solve_flow(mesh_file  = name,
                                                final_time = self.final_time,
                                                reynolds   = self.reynolds,
                                                output     = self.output,
                                                cfl        = self.cfl,
                                                pts_x      = env.shape.control_pts[:,0],
                                                pts_y      = env.shape.control_pts[:,1],
                                                xmin       = self.xmin,
                                                xmax       = self.xmax,
                                                ymin       = self.ymin,
                                                ymax       = self.ymax)
                # Save solution png
                os.system('mv '+str(env.shape.index)+'.png '+self.comp_dir+'/save/sol/.')
            except Exception as exc:
                print(exc)
                solved = False

            # If solver was successful
            if (solved):
                # pressure_drop is always <0 while heat changes sign
                penal  = 0.0
                # heat   =-heat # Make heat positive
                #if (heat > 2.0): heat=2.0*heat # Shaping for faster convergence

                reward = 2* heat/(abs(pressure_drop)**(1/3)*1000)
                reward = max(reward, -10.0)

            # If solver was not successful
            else:
                pressure_drop   =-1.0
                heat   = 0.0
                reward =-5.0
                penal  = 5.0

        # If meshing was not successful, we just return a high penalization
        else:
            pressure_drop   =-1.0
            heat   = 0.0
            reward =-5.0
            penal  = 5.0

        # Save pressure_drop, heat, reward and penalization
        self.pressure_drop   = np.append(self.pressure_drop,   pressure_drop)
        self.heat   = np.append(self.heat,   heat)
        self.reward = np.append(self.reward, reward)
        self.penal  = np.append(self.penal,  penal)

        val_pressure_drop   = np.sum(self.pressure_drop)/env.shape.index
        val_heat   = np.sum(self.heat)/env.shape.index
        val_reward = np.sum(self.reward)/env.shape.index
        self.avg_pressure_drop   = np.append(self.avg_pressure_drop,   val_pressure_drop)
        self.avg_heat   = np.append(self.avg_heat,   val_heat)
        self.avg_reward = np.append(self.avg_reward, val_reward)

    def save_qoi(self):
        # Retrieve current index
        i = env.shape.index

        # Write pressure_drop/heat values to file
        filename = self.comp_dir+'/save/pressure_drop_heat'
        with open(filename, 'a') as f:
            f.write('{} {} {} {} {}\n'.format(i,
                                              self.pressure_drop[-1],
                                              self.heat[-1],
                                              self.avg_pressure_drop[-1],
                                              self.avg_heat[-1]))

        # Write reward and penalization to file
        filename = self.comp_dir+'/save/reward_penalization'
        with open(filename, 'a') as f:
            f.write('{} {} {} {}\n'.format(i,
                                           self.reward[-1],
                                           self.penal[-1],
                                           self.avg_reward[-1]))

    def fill_next_state(self, meshed, index):
        next_state_0 = np.array([])
        for i in range(0,env.shape.n_control_pts):
            next_state_0 = np.append(next_state_0,env.shape.control_pts[i,0])
            next_state_0 = np.append(next_state_0,env.shape.control_pts[i,1])
            next_state_0 = np.append(next_state_0,env.shape.edgy[i])

        return next_state_0

    def fill_next_state_1(self, meshed, index):
        next_state_1 = np.array([])
        for i in range(0,env.shape_1.n_control_pts_1):
            next_state_1 = np.append(next_state_1,env.shape_1.control_pts_1[i,0])
            next_state_1 = np.append(next_state_1,env.shape_1.control_pts_1[i,1])
            next_state_1 = np.append(next_state_1,env.shape_1.edgy_1[i])

        return next_state_1
    
    def fill_next_state_2(self, meshed, index):
        next_state_2 = np.array([])
        for i in range(0,env.shape_2.n_control_pts_2):
            next_state_2 = np.append(next_state_2,env.shape_2.control_pts_2[i,0])
            next_state_2 = np.append(next_state_2,env.shape_2.control_pts_2[i,1])
            next_state_2 = np.append(next_state_2,env.shape_2.edgy_2[i])

        return next_state_2

    def fill_next_state_3(self, meshed, index):
        next_state_3 = np.array([])
        for i in range(0,env.shape_3.n_control_pts_3):
            next_state_3 = np.append(next_state_3,env.shape_3.control_pts_3[i,0])
            next_state_3 = np.append(next_state_3,env.shape_3.control_pts_3[i,1])
            next_state_3 = np.append(next_state_3,env.shape_3.edgy_3[i])

        return next_state_3

    def fill_next_state_4(self, meshed, index):
        next_state_4 = np.array([])
        for i in range(0,env.shape_4.n_control_pts_4):
            next_state_4 = np.append(next_state_4,env.shape_4.control_pts_4[i,0])
            next_state_4 = np.append(next_state_4,env.shape_4.control_pts_4[i,1])
            next_state_4 = np.append(next_state_4,env.shape_4.edgy_4[i])

        return next_state_4
    
    
