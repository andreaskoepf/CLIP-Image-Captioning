import json
import csv
from pathlib import Path
import torch
import matplotlib.pyplot as plt
import numpy as np


def plot_histogram(data, label, title, x_label='Word count', y_label='Frequency'):
    plt.figure(figsize=(8,6))
    plt.hist(data, alpha=1.0, label=label)
    plt.xlabel(x_label, size=14)
    plt.ylabel(y_label, size=14)
    plt.title(title)
    plt.legend(loc='upper right')


def process_folder(folder_name):
    folder_path = Path(folder_name)
    gt_path = folder_path / 'gt.csv'

    human_captions = []
    synthetic_captions = []
    
    with open(gt_path, newline='') as f:
        reader = csv.reader(f)
        headers = next(reader)
        for row in reader:
            if row[2] == 'True':
                human_captions.append(row[3])
            else:
                synthetic_captions.append(row[4])

    human_lengths = [len(s.split(' ')) for s in human_captions]
    synth_lengths = [len(s.split(' ')) for s in synthetic_captions]

    plot_histogram([human_lengths, synth_lengths], ['Human', 'Synthetic'], title=f'Captions Lengths Histogram {folder_name}')
    fn = f'word_count_hist_{folder_name}.'
    plt.savefig(fn + 'png')
    plt.savefig(fn + 'svg')


def main():
    #folders = ['eval_A']
    folders = ['eval_A', 'eval_B', 'eval_C', 'eval_D', 'eval_E', 'eval_F', 'eval_X1', 'eval_X2', 'eval_X3', 'eval_X4', 'eval_Y1', 'eval_Y2']

    for folder_name in folders:
        process_folder(folder_name)


if __name__ == '__main__':
    main()