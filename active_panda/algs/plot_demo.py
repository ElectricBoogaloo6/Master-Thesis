import numpy as np
import os
import argparse
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


CURRENT_DIR = os.getcwd()
PARENT_DIR = os.path.dirname(CURRENT_DIR)



def recover_demo_trajs(queried_inits, demo_pool, init_state_pool, task_name):
    recover_trajs = []

    total_query_num = queried_inits.shape[0]
    for i in range(total_query_num):
        queried_init = queried_inits[i]
        dist_list = []
        for init_state in init_state_pool:
            if task_name == 'ReachWithObstacleV0' or task_name == 'ReachWithObstacleV1':
                goal_position = init_state[-3:]
                dist = np.linalg.norm(queried_init - goal_position)
                dist_list.append(dist)
            elif task_name == 'PushWithObstacleV0' or task_name == 'PushWithObstacleV1':
                object_position = init_state[6:9]
                dist = np.linalg.norm(queried_init - object_position)
                dist_list.append(dist)


        recover_demo_id = np.argmin(np.array(dist_list))
        recover_traj = demo_pool[recover_demo_id].copy()
        recover_trajs.append(recover_traj)

        print("Recover demo: {}, dist: {}".format(i + 1, dist_list[recover_demo_id]))
        # print(init_state_pool[recover_demo_id])

    return recover_trajs


def cuboid_data(o, size=(1, 1, 1)):
    # code taken from
    # https://stackoverflow.com/a/35978146/4124317
    # suppose axis direction: x: to left; y: to inside; z: to upper
    # get the length, width, and height
    l, w, h = size
    x = [[o[0], o[0] + l, o[0] + l, o[0], o[0]],
         [o[0], o[0] + l, o[0] + l, o[0], o[0]],
         [o[0], o[0] + l, o[0] + l, o[0], o[0]],
         [o[0], o[0] + l, o[0] + l, o[0], o[0]]]
    y = [[o[1], o[1], o[1] + w, o[1] + w, o[1]],
         [o[1], o[1], o[1] + w, o[1] + w, o[1]],
         [o[1], o[1], o[1], o[1], o[1]],
         [o[1] + w, o[1] + w, o[1] + w, o[1] + w, o[1] + w]]
    z = [[o[2], o[2], o[2], o[2], o[2]],
         [o[2] + h, o[2] + h, o[2] + h, o[2] + h, o[2] + h],
         [o[2], o[2], o[2] + h, o[2] + h, o[2]],
         [o[2], o[2], o[2] + h, o[2] + h, o[2]]]
    return np.array(x), np.array(y), np.array(z)


def plotCubeAt(pos=(0, 0, 0), size=(1, 1, 1), ax=None, **kwargs):
    # Plotting a cube element at position pos
    if ax != None:
        X, Y, Z = cuboid_data(pos, size)
        ax.plot_surface(X, Y, Z, rstride=1, cstride=1, **kwargs, shade=False, zorder=0)



def build_scene(task_name, ax):
    if task_name == 'ReachWithObstacleV0':
        center_positions = [(-0.05, -0.1, 0.075)]
        sizes = [(0.02, 0.2, 0.15)]
    elif task_name == 'ReachWithObstacleV1':
        center_positions = [(-0.05, -0.1, 0.075), (0.0, 0.08, 0.09)]
        sizes = [(0.02, 0.2, 0.15), (0.05, 0.05, 0.18)]
    elif task_name == 'PushWithObstacleV0':
        center_positions = [(-0.11, 0.0, 0.01)]
        sizes = [(0.02, 0.1, 0.02)]
    elif task_name == 'PushWithObstacleV1':
        center_positions = [(-0.05, 0.01, 0.01), (0.05, -0.1, 0.01)]
        sizes = [(0.01, 0.1, 0.02), (0.01, 0.1, 0.02)]


    for cp, s in zip(center_positions, sizes):
        cp = np.array(cp)
        s = np.array(s)
        p = cp - 0.5 * s

        plotCubeAt(pos=p, size=s, ax=ax, color="lightgrey", edgecolor="k")



def main():
    seed = 6
    np.random.seed(seed)

    task_names = ['ReachWithObstacleV0', 'ReachWithObstacleV1', 'PushWithObstacleV0', 'PushWithObstacleV1']
    methods = ['ours', 'early', 'no_bc', 'no_warmup']

    for task_name in task_names:
        if task_name == 'ReachWithObstacleV0':
            max_demo_num = 20
            warm_up_demo_num = 5
        elif task_name == 'ReachWithObstacleV1':
            max_demo_num = 20
            warm_up_demo_num = 10
        elif task_name == 'PushWithObstacleV0':
            max_demo_num = 20
            warm_up_demo_num = 10
        elif task_name == 'PushWithObstacleV1':
            max_demo_num = 60
            warm_up_demo_num = 20

        for method_id in range(len(methods)):
            method_shortname = methods[method_id]
            if method_shortname == 'ours':
                method = 'active_ddpgfd_bc_warmstart_transfer_learning'
                warmup = True
            elif method_shortname == 'early':
                method = 'early_ddpg_transfer_learning'
                warmup = False
            elif method_shortname == 'no_bc':
                method = 'active_ddpgfd_no_bc_transfer_learning'
                warmup = True
            elif method_shortname == 'no_warmup':
                method = 'active_ddpgfd_no_warmup_transfer_learning'
                warmup = False


            # load the pool of demonstrations
            joystick_demo_path = PARENT_DIR + '/demo_data/joystick_demo/' + task_name + '/'
            demo_state_trajs = np.genfromtxt(joystick_demo_path + 'demo_state_trajs.csv', delimiter=' ')
            starting_ids = []
            for i in range(demo_state_trajs.shape[0]):
                if demo_state_trajs[i][0] == np.inf:
                    starting_ids.append(i)

            demo_pool = []
            init_state_pool = []
            total_demo_num = len(starting_ids)
            for i in range(total_demo_num):
                demo_start_step = starting_ids[i] + 1
                if i < total_demo_num - 1:
                    demo_end_step = starting_ids[i + 1]
                else:
                    demo_end_step = demo_state_trajs.shape[0]

                demo = demo_state_trajs[demo_start_step:demo_end_step, :]
                demo_pool.append(demo)

                init_state = demo_state_trajs[demo_start_step, :]
                init_state_pool.append(init_state)


            # load queried initial states
            queried_demo_path = PARENT_DIR + '/' + 'demo_data/visualization/' + task_name + '/' + method + '/max_demo_' + str(
                max_demo_num) + '/transfer/'
            if task_name == 'ReachWithObstacleV0' or task_name == 'ReachWithObstacleV1':
                queried_demo_path += 'demo_goal_positions.csv'
            elif task_name == 'PushWithObstacleV0' or task_name == 'PushWithObstacleV1':
                queried_demo_path += 'demo_object_positions.csv'
            queried_inits = np.genfromtxt(queried_demo_path, delimiter=' ')

            # recover queried demo trajectories
            warmup_demo_ids = np.random.choice(max_demo_num, warm_up_demo_num, replace=False)
            warmup_demos = []
            for id in warmup_demo_ids:
                warmup_demo = demo_pool[id].copy()
                warmup_demos.append(warmup_demo)
            recover_queried_demos = recover_demo_trajs(queried_inits=queried_inits,
                                                       demo_pool=demo_pool,
                                                       init_state_pool=init_state_pool,
                                                       task_name=task_name)

            # build the scene
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            # ax.set_aspect('equal', zoom=2.0)
            ax.set_box_aspect((1, 1, 0.6), zoom=1.5)

            if task_name == 'ReachWithObstacleV0' or task_name == 'ReachWithObstacleV1':
                ax.view_init(elev=16, azim=-146)
            elif task_name == 'PushWithObstacleV0' or task_name == 'PushWithObstacleV1':
                ax.view_init(elev=16, azim=-115)


            build_scene(task_name=task_name, ax=ax)
            # build_scene_2(task_name=task_name, ax=ax)

            # plot queried demos
            if warmup:
                for i in range(len(warmup_demos)):
                    demo = warmup_demos[i]
                    xs = demo[:, 0]
                    ys = demo[:, 1]
                    zs = demo[:, 2]
                    ax.plot3D(xs, ys, zs, 'darkorange', alpha=0.5, zorder=6)

            alpha_list = np.linspace(0.5, 1.0, len(recover_queried_demos))
            for i in range(len(recover_queried_demos)):
                demo = recover_queried_demos[i]
                xs = demo[:, 0]
                ys = demo[:, 1]
                zs = demo[:, 2]
                ax.plot3D(xs, ys, zs, 'green', alpha=alpha_list[i], zorder=6)


            ax.set_xlim(-0.25, 0.25)
            ax.set_ylim(-0.25, 0.25)
            ax.set_zlim(0.0, 0.3)

            plt.savefig(PARENT_DIR + '/plots/' + 'demo_' + task_name + '_' + method_shortname + '.png', dpi=300)
            plt.show()



if __name__ == '__main__':
   main()