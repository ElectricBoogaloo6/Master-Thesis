import numpy as np
import os
import argparse
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

CURRENT_DIR = os.getcwd()
PARENT_DIR = os.path.dirname(CURRENT_DIR)



def plot_timing(timing_df, hue, task_name):
    sns.set_style('darkgrid')
    l = 10
    h_l_ratio = 0.8
    h = l * h_l_ratio
    sns.set_theme(rc={'figure.figsize': (l, h)})

    plt.figure(figsize=(l, h))
    colors = ["#EA8D9E", '#8694AD', '#FEBE67', '#C85D3D']


    f = sns.lineplot(data=timing_df, x="step", y="total_demo_num", hue=hue,
                 palette=sns.color_palette(colors), marker="o")
    f.set(xlabel='environment steps (× 1000)', ylabel='total demos num')

    if task_name == 'PushWithObstacleV1':
        ticks = np.arange(0, 61, 5)
    else:
        ticks = np.arange(0, 21, 2)
    f.axes.set_yticks(ticks)

    f.axes.xaxis.set_tick_params(labelsize=20)
    f.axes.yaxis.set_tick_params(labelsize=20)
    f.axes.xaxis.label.set_size(20)
    f.axes.yaxis.label.set_size(20)


    # plt.legend(ncol=4, loc='lower center', framealpha=1.0, bbox_to_anchor=(0.5, -0.17), facecolor='white',
    #            edgecolor='white', fontsize=20)

    plt.legend(fontsize=20)
    plt.tight_layout()

    plt.savefig(PARENT_DIR + '/plots/' + 'timing_' + task_name + '.png', dpi=300)
    plt.show()


def main():
    task_names = ['ReachWithObstacleV0', 'ReachWithObstacleV1', 'PushWithObstacleV0', 'PushWithObstacleV1']
    methods = ['ours(no-BC)', 'ours(no-warmup)', 'EARLY', 'ours']
    original_method_names = ['no_bc',
                             'no_warmup',
                             'early',
                             'ours']

    sns.set_style('darkgrid', {'font.family': 'serif', 'font.serif': 'Times New Roman'})

    for task_id in range(len(task_names)):
        task_name = task_names[task_id]
        timing_data_dict = {'step': [], 'total_demo_num': [], 'method':[]}

        for method_id in range(len(methods)):
            method = methods[method_id]
            original_saving_name = original_method_names[method_id]

            data_path = PARENT_DIR + '/demo_data/timing/' + task_name + '/' + original_saving_name + '.csv'
            timing_data = np.genfromtxt(data_path, delimiter=',')
            for i in range(timing_data.shape[0]):
                step = timing_data[i][0]
                total_demo_num = timing_data[i][1]
                timing_data_dict['step'].append(step/1000.0)
                timing_data_dict['total_demo_num'].append(total_demo_num)
                timing_data_dict['method'].append(method)

        timing_df = pd.DataFrame(data=timing_data_dict)
        plot_timing(timing_df=timing_df, hue='method', task_name=task_name)


if __name__ == '__main__':
    main()


