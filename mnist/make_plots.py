import matplotlib.pyplot as plt
import seaborn as sns
from csv_utils import load_csv
import os

def plot_dice():
    # First plot dice over epochs for both training and validation.
    plt.figure(figsize=(16, 9))
    clrs = sns.color_palette("husl", 5)
    with sns.axes_style('white'):
        plt.grid()
        run = '00'
        net = 'group'
        plt.xticks(range(20))
        plt.ylim([80, 100])
        log_file = f'logs/cnn_{net}_{run}.csv' 
        (epochs, accs, _, _) = load_csv(log_file,
                                        ['epoch', 'test_accuracy',
                                        'start_time', 'cur_time'],
                                        [int, float, float, float])
        plt.ylabel('accuracy %')
        plt.xlabel('epoch')
        p = plt.plot(epochs, accs, label=f'test accuracy {net} {run}, max: {round(max(accs), 3)}')
        plt.legend()

    fig_path = os.path.join('plots', f'{net}_{run}.png')
    print('saving figure to ', fig_path)
    plt.savefig(fig_path)


if __name__ == '__main__':
    plot_dice()


