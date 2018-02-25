import numpy as np
import matplotlib.pyplot as plt
from promp_ctrlr import PROMPCtrl
from promps import DiscretePROMP


np.random.seed(0)

# #load the data
# main_traj = np.sin(np.linspace(-np.pi, np.pi, 201)).tolist()

# traj_list = []

# for _ in range(10):
#     traj_list.append(np.random.normal(main_traj,0.01))



def plot_mean_and_sigma(mean, sigma, interval=3, color_mean=None, color_shading=None, label=''):

    """
    Expects mean = [Nx1] and Sigma [NxN]
    The computed sigma is NXN, so diag of that is taken to get the variance
    """

    lower_bound = mean - interval*np.diag(sigma)
    upper_bound = mean + interval*np.diag(sigma)

    # plot the shaded range of the confidence intervals
    plt.fill_between(range(mean.shape[0]), lower_bound, upper_bound, color=color_shading, alpha=.5)
    # plot the mean on top
    plt.plot(mean, color_mean, label=label)

    ########################to remove duplicate label handles########################
    handles, labels = plt.gca().get_legend_handles_labels()
    i =1
    while i<len(labels):
        if labels[i] in labels[:i]:
            del(labels[i])
            del(handles[i])
        else:
            i +=1

    plt.legend(handles, labels)
    ###############################################################################


def generate_trajectories_from_dynamics(num_trajs = 100, A = np.array([ [0.,1.], [0., 0.] ]), B = np.array([ [0.], [1.] ]), start_x = np.array([[0],[0]]), noise = 1):

    num_points = 200 # same as promps phase time_steps
    '''
    x = [pos vel].T
    '''

    traj_list = []


    def next_x(prev_x, cmd):
        dt = 0.005 # same as promps dt
        return np.dot(np.eye(2) + A*dt, prev_x) + np.dot(B*dt, cmd)

    demo_list = []
    ddemo_list = []

    for _ in range(num_trajs):

        prev_x = start_x

        pos_traj = np.zeros(num_points)
        vel_traj = np.zeros(num_points)

        pos_traj[0] = start_x[0]
        vel_traj[0] = start_x[1]


        cmd_list = np.sin(np.linspace(0,4*np.pi,num_points))
        cmd = cmd_list + np.random.randn(num_points)*0.1
        for i in range(1,num_points):

            x = next_x(prev_x, cmd[i-1])
            pos_traj[i] = x[0]
            vel_traj[i] = x[1]
            prev_x = x
        # plt.plot(pos_traj)
        # plt.show()
        # raw_input()
        demo_list.append(pos_traj)
        ddemo_list.append(vel_traj)
        # cmd_list.append(cmd)

    return demo_list, ddemo_list, cmd_list


def demo_generate_traj(additional_viapoint=False):
    """
    Make the additional_viapoint False to 
    see the control commands accurately reproduced
    by the promp. The conditioning takes it harder whe
    additional via points are added
    """

    #set the start and goal, the spatial scaling
    # d_promp.set_start(demos_list[0][0])
    # d_promp.set_goal(demos_list[0][-1])


    if additional_viapoint:
        #add a via point
        d_promp.add_viapoint(0.2, .02)
        d_promp.add_viapoint(0.7, .04)

    traj_data_1 = d_promp.generate_trajectory(phase_speed=0.8,  randomness=1e-1)
    traj_data_2 = d_promp.generate_trajectory(phase_speed=1.,   randomness=1e-1)
    traj_data_3 = d_promp.generate_trajectory(phase_speed=1.33, randomness=1e-1)

    # print d_promp._Ddemo_trajs
    for traj, traj_vel in zip(demo_trajs, ddemo_trajs):
        """
        This labeling in the loop causes
        multiple labels to be added to the axis
        Is there a way to avoid it than the funciton
        given in plot_mean_and_sigma above?
        """

        plt.figure("ProMP-Pos")
        plt.plot(traj, 'k', alpha=0.2, label='demo_pos')

        plt.figure("ProMP-Vel")
        plt.plot(traj_vel, 'k', alpha=0.2, label='demo_vel')


    plt.figure("ProMP-Pos")

    plot_mean_and_sigma(mean=traj_data_1['mu_traj'].squeeze(), sigma=traj_data_1['sigma_traj'], color_mean='r', color_shading='r', label='speed=0.8')
    plot_mean_and_sigma(mean=traj_data_2['mu_traj'].squeeze(), sigma=traj_data_2['sigma_traj'], color_mean='g', color_shading='g', label='speed=1.')
    plot_mean_and_sigma(mean=traj_data_3['mu_traj'].squeeze(), sigma=traj_data_3['sigma_traj'], color_mean='b', color_shading='b', label='speed=1.33')

    if additional_viapoint:
        """
        Though phase of each of these traj_data is
        different, the via point will be present in the 
        same phase on all, so it is okay to take the default phase
        to find the plot location
        """
        for viapoint in d_promp._viapoints:
            #plot only additional viapoints
            if (viapoint['t'] > 0) and (viapoint['t'] < 1):
                # print viapoint['t']
                time_step = d_promp._phase.get_time_step(viapoint['t'])
                print time_step , "t"
                plt.scatter(time_step, viapoint['traj_point'], marker='*', s=500)


    plt.figure("ProMP-Vel")

    plot_mean_and_sigma(mean=traj_data_1['mu_Dtraj'].squeeze(), sigma=traj_data_1['sigma_Dtraj'], color_mean='r', color_shading='r', label='speed=0.8')
    plot_mean_and_sigma(mean=traj_data_2['mu_Dtraj'].squeeze(), sigma=traj_data_2['sigma_Dtraj'], color_mean='g', color_shading='g', label='speed=1.')
    plot_mean_and_sigma(mean=traj_data_3['mu_Dtraj'].squeeze(), sigma=traj_data_3['sigma_Dtraj'], color_mean='b', color_shading='b', label='speed=1.33')


def compute_ctrl_cmds(traj_data, color='k', label='', original_actions=None):

    A = np.array([ [0.,1.], [0., 0.] ])
    B = np.array([ [0.], [1.] ])

    promp_ctl = PROMPCtrl(traj_data=traj_data)
    promp_ctl.update_system_matrices(A=A, B=B)

    plt.figure("Ctrl cmds")

    state_list = np.hstack([traj_data['mu_traj'],  traj_data['mu_Dtraj']]) #data['states'][0] #

    ctrl_cmds_mean, ctrl_cmds_sigma = promp_ctl.compute_ctrl_traj(state_list=state_list)

    plot_mean_and_sigma(mean=ctrl_cmds_mean[:, 0], sigma=np.diag(ctrl_cmds_sigma[:, 0, 0]), color_mean=color, color_shading=color, label=label)

    if original_actions is not None:

        plt.plot(original_actions, 'k', label='original control')

    plt.legend()


def create_ctrl_traj():
    """
    This funciton shows how to compute 
    closed form control distribution from the trajectory distribution
    """
    # action = data['actions'][0]
    action = cmd_list

    traj_data_1 = d_promp.generate_trajectory(phase_speed=0.8,  randomness=1e-1)
    traj_data_2 = d_promp.generate_trajectory(phase_speed=1.,   randomness=1e-1)
    traj_data_3 = d_promp.generate_trajectory(phase_speed=1.33, randomness=1e-1)

    compute_ctrl_cmds(traj_data_1, color='r', label='speed=0.8', original_actions=None)
    #only we know how the mean of speed 1 trajectory was produced
    compute_ctrl_cmds(traj_data_2, color='g', label='speed=1.', original_actions=action)
    
    compute_ctrl_cmds(traj_data_3, color='b', label='speed=1.33', original_actions=None)

def main():


    demo_generate_traj(True)
    create_ctrl_traj()
    plt.show()


if __name__ == '__main__':
    demo_trajs, ddemo_trajs, cmd_list = generate_trajectories_from_dynamics()
    d_promp = DiscretePROMP(data=demo_trajs)
    d_promp.train()
    main() 


