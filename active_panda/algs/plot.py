import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import os


CURRENT_DIR = os.getcwd()
PARENT_DIR = os.path.dirname(CURRENT_DIR)



def plot_success_rate(success_df, hue, plot_name):
    sns.set_style('darkgrid')

    # sns.set_theme(rc={'figure.figsize': (11.7, 8.27)})
    sns.set_theme(rc={'figure.figsize': (10, 8)})
    # plt.rcParams["figure.figsize"] = (10, 7)

    colors = ["#f5852b", "#6d9855", "#3988b9", '#ee2c4d']
    labels = ['DDPGfD-BC', 'DDPG-LfD', 'EARLY', 'ours']

    # plot average episode success rate
    f1 = sns.lineplot(data=success_df, x="step", y="success_rate", hue=hue, errorbar=('ci', 95),
                      palette=sns.color_palette(colors))
    f1.set(xlabel='environment steps (× 1000)', ylabel='average success rate')


    plt.ylim(-0.05, 1.05)

    plt.legend(ncol=4, loc='lower center', framealpha=1.0, bbox_to_anchor=(0.5, -0.15), facecolor='white', edgecolor='white')

    # plt.savefig(PARENT_DIR + '/plots/' + 'success_rate_' + plot_name + '.png', dpi=300)
    plt.show()


def subplot_success_rate(success_df, hue, ax, scenario_name):
    # colors = ['#4A6375', '#6CAA89', '#FEBE67', '#C85D3D'] # with green
    # colors = ['#4A6375', '#7FA5B7', '#FEBE67', '#C85D3D'] # with pale blue
    colors = ['#4A6375', '#9A8870', '#FEBE67', '#C85D3D']

    f = sns.lineplot(data=success_df, x="step", y="success_rate", hue=hue, errorbar=('ci', 95),
                 palette=sns.color_palette(colors), ax=ax)
    f.set(xlabel='environment steps (× 1000)', ylabel='average success rate')

    ax.get_legend().remove()
    ax.set_ylim([-0.05, 1.05])
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

    ax.yaxis.set_tick_params(labelbottom=True, labelsize=20)
    ax.xaxis.set_tick_params(labelsize=20)

    ax.xaxis.label.set_size(20)
    ax.yaxis.label.set_size(20)

    ax.set_title(scenario_name, fontsize = 20)


def subplot_success_rate_ablation(success_df, hue, ax, scenario_name):
    colors = ["#EA8D9E", '#8694AD', '#FEBE67', '#C85D3D']


    f = sns.lineplot(data=success_df, x="step", y="success_rate", hue=hue, errorbar=('ci', 95),
                 palette=sns.color_palette(colors), ax=ax)
    f.set(xlabel='environment steps (× 1000)', ylabel='average success rate')

    ax.get_legend().remove()
    ax.set_ylim([-0.05, 1.05])
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

    ax.yaxis.set_tick_params(labelbottom=True, labelsize=20)
    ax.xaxis.set_tick_params(labelsize=20)

    ax.xaxis.label.set_size(20)
    ax.yaxis.label.set_size(20)

    ax.set_title(scenario_name, fontsize = 20)


def smooth(data, sm=1):
    smooth_data = []
    if sm > 1:
        for d in data:
            z = np.ones(len(d))
            y = np.ones(sm)*1.0
            d = np.convolve(y, d, "same")/np.convolve(y, z, "same")
            smooth_data.append(d)
    else:
        smooth_data = data
    return smooth_data


def smooth_new(scalars, weight):  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)  # Save it
        last = smoothed_val  # Anchor the last smoothed value

    return smoothed


def main():
    methods = ['DDPGfD-BC', 'DDPG-LfD', 'EARLY', 'ours']
    original_method_names = ['ddpgfd_bc_transfer_learning',
                             'dddpglfd-transfer-learning',
                             'early_ddpg_transfer_learning',
                             'active_ddpgfd_bc_warmstart_transfer_learning']
    task_names = ['ReachWithObstacleV0', 'ReachWithObstacleV1', 'PushWithObstacleV0', 'PushWithObstacleV1']
    scenario_names = ['Reach2ReachWithObs-V1', 'Reach2ReachWithObs-V2', 'Push2PushWithObs-V1', 'Push2PushWithObs-V2']

    sns.set_style('darkgrid', {'font.family':'serif', 'font.serif':'Times New Roman'})
    # fig, axes = plt.subplots(1, 4, gridspec_kw={'wspace': 0.3, 'bottom': 0.3}, figsize=(10*1.4*10, 3.2), sharey=True)
    h = 5
    length_to_height_ratio = 1.8
    l = h * length_to_height_ratio * 3
    fig, axes = plt.subplots(1, 4, gridspec_kw={'wspace': 0.3, 'bottom': 0.3, "left":0.05, "right":0.98}, figsize=(l, h),
                             sharey=True)

    lines = []


    for task_id in range(len(task_names)):
        task_name = task_names[task_id]
        step_success_data_dict = {'step': [], 'success_rate': [], 'method': []}
        scenario_name = scenario_names[task_id]

        if task_name == 'PushWithObstacleV1':
            max_demo_num = 60
        else:
            max_demo_num = 20

        for method_id in range(len(methods)):
            method = methods[method_id]
            original_saving_name = original_method_names[method_id]

            res_path_list = []
            res_path = PARENT_DIR + '/evaluation_res/' + task_name + '/' + original_saving_name + '/max_demo_' + str(
                        max_demo_num) + '/transfer/'
            res_path_list.append(res_path)

            for res_path in res_path_list:
                success_res_per_step = np.genfromtxt(res_path + 'success_res_per_step.csv', delimiter=' ')
                for seed_id in range(1, success_res_per_step.shape[1]):
                    all_steps_data = success_res_per_step[:, seed_id]
                    smoothed_data = np.array(smooth_new(all_steps_data, 0.8))
                    for j in range(smoothed_data.shape[0]):
                        step = success_res_per_step[j][0]
                        smoothed_success = smoothed_data[j]
                        step_success_data_dict['step'].append(int(step / 1000))
                        step_success_data_dict['success_rate'].append(smoothed_success)
                        step_success_data_dict['method'].append(method)

        step_success_df = pd.DataFrame(data=step_success_data_dict)
        # plot_success_rate(success_df=step_success_df, hue='method', plot_name=task_name)

        subplot_success_rate(success_df=step_success_df, hue='method', ax=axes[task_id], scenario_name=scenario_name)


    lines, labels = axes[-1].get_legend_handles_labels()
    fig.legend(lines, labels, ncol=4, loc='lower center', framealpha=1.0, bbox_to_anchor=(0.5, 0.001), facecolor='white', edgecolor='white', fontsize=20)

    # fig.savefig(PARENT_DIR + '/plots/' + 'success_rate_res' + '.png')

    plt.show()


def main_ablation():
    methods = ['ours(no-BC)', 'ours(no-warmup)', 'EARLY', 'ours']
    original_method_names = ['active_ddpgfd_no_bc_transfer_learning',
                             'active_ddpgfd_no_warmup_transfer_learning',
                             'early_ddpg_transfer_learning',
                             'active_ddpgfd_bc_warmstart_transfer_learning']
    task_names = ['ReachWithObstacleV0', 'ReachWithObstacleV1', 'PushWithObstacleV0', 'PushWithObstacleV1']
    scenario_names = ['Reach2ReachWithObs-V1', 'Reach2ReachWithObs-V2', 'Push2PushWithObs-V1', 'Push2PushWithObs-V2']

    sns.set_style('darkgrid', {'font.family': 'serif', 'font.serif': 'Times New Roman'})
    # fig, axes = plt.subplots(1, 4, gridspec_kw={'wspace': 0.3, 'bottom': 0.3}, figsize=(10*1.4*10, 3.2), sharey=True)
    """h = 10
    length_to_height_ratio = 1.1
    l = h * length_to_height_ratio
    fig, axes = plt.subplots(2, 2, gridspec_kw={'wspace': 0.23, 'hspace': 0.55, 'bottom': 0.16, "left": 0.1, "right": 0.98},
                             figsize=(l, h), sharey=True)"""

    h = 5
    length_to_height_ratio = 1.8
    l = h * length_to_height_ratio * 3
    fig, axes = plt.subplots(1, 4, gridspec_kw={'wspace': 0.3, 'bottom': 0.3, "left": 0.05, "right": 0.98},
                             figsize=(l, h),
                             sharey=True)


    lines = []

    for task_id in range(len(task_names)):
        task_name = task_names[task_id]
        step_success_data_dict = {'step': [], 'success_rate': [], 'method': []}
        scenario_name = scenario_names[task_id]

        if task_name == 'PushWithObstacleV1':
            max_demo_num = 60
        else:
            max_demo_num = 20

        for method_id in range(len(methods)):
            method = methods[method_id]
            original_saving_name = original_method_names[method_id]

            res_path_list = []
            res_path = PARENT_DIR + '/evaluation_res/' + task_name + '/' + original_saving_name + '/max_demo_' + str(
                max_demo_num) + '/transfer/'
            res_path_list.append(res_path)

            for res_path in res_path_list:
                success_res_per_step = np.genfromtxt(res_path + 'success_res_per_step.csv', delimiter=' ')
                for seed_id in range(1, success_res_per_step.shape[1]):
                    all_steps_data = success_res_per_step[:, seed_id]
                    smoothed_data = np.array(smooth_new(all_steps_data, 0.8))
                    for j in range(smoothed_data.shape[0]):
                        step = success_res_per_step[j][0]
                        smoothed_success = smoothed_data[j]
                        step_success_data_dict['step'].append(int(step / 1000))
                        step_success_data_dict['success_rate'].append(smoothed_success)
                        step_success_data_dict['method'].append(method)

        step_success_df = pd.DataFrame(data=step_success_data_dict)
        # plot_success_rate(success_df=step_success_df, hue='method', plot_name=task_name)

        """if task_id == 0:
            ax = axes[0][0]
        elif task_id == 1:
            ax = axes[0][1]
        elif task_id == 2:
            ax = axes[1][0]
        else:
            ax = axes[1][1]
        """

        ax = axes[task_id]


        subplot_success_rate_ablation(success_df=step_success_df, hue='method', ax=ax, scenario_name=scenario_name)

    # lines, labels = axes[1][1].get_legend_handles_labels()
    lines, labels = axes[-1].get_legend_handles_labels()
    fig.legend(lines, labels, ncol=4, loc='lower center', framealpha=1.0, bbox_to_anchor=(0.5, 0.001),
               facecolor='white', edgecolor='white', fontsize=20)

    # fig.savefig(PARENT_DIR + '/plots/' + 'success_rate_res' + '.png')

    plt.show()


if __name__=='__main__':
    main()
    # main_ablation()

