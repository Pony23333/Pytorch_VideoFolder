import os
import os.path
import matplotlib.pyplot as plt
import numpy as np
VIDEO_EXTENSIONS = ('.mp4', 'avi', 'rmvb')


def is_video_file(filename):
    '''To judge if a file is video'''
    return any(filename.endswith(extension) for extension in VIDEO_EXTENSIONS)


class DatasetStatistics:
    '''
    This class is for getting insight of a dataset. Given the root path of the dataset, this class will
    record statistics features of the given dataset.

    self.class_to_idx: the mapping from class name to index
    self.idx_to_class: the mapping from index to class name
    self.class_count: count of each class. Stored in a dictionary where key is the index
    '''
    def __init__(self, root_path):
        self.root_path = root_path
        self.dataset = {}
        self.class_name_to_idx()
        self.make_stat()

    def class_name_to_idx(self):
        '''Make a mapping from class name to index'''
        classes = [d for d in os.listdir(self.root_path) if os.path.isdir(os.path.join(self.root_path, d))]
        classes.sort()
        self.class_to_idx = {classes[i]: i for i in range(len(classes))}
        self.idx_to_class = { value:key for key,value in self.class_to_idx.items()}


    def make_stat(self):

        dir = os.path.expanduser(self.root_path)  # expand the user's home directory to a absolute path
        for target in sorted(os.listdir(dir)):
            d = os.path.join(dir, target)
            if not os.path.isdir(d):
                continue
            self.dataset[self.class_to_idx[target]] = 0
            for root, _, fnames in sorted(os.walk(d)):
                for fname in fnames:
                    if is_video_file(fname):
                        self.dataset[self.class_to_idx[target]] += 1

        index, self.class_count = zip(*self.dataset.items())
        indices = sorted(range(len(self.class_count)), key=lambda k: self.class_count[k], reverse=True)
        self.class_count = np.array(self.class_count)[indices]
        index = np.array(index)[indices]
        self.class_name = [self.idx_to_class[i] for i in index]
        self.total = self.class_count.sum()


    def bar_chart_top(self, top=10, path='bar_chart.png'):
        assert top <= len(self.class_name)

        y_pos = range(top,0,-1)
        plt.barh(y_pos, self.class_count[:top], align='center', alpha=0.5)
        plt.yticks(y_pos, self.class_name[:top])
        plt.xlabel('class name')
        plt.title('Statistic of the top {} class in the dataset'.format(top))
        plt.savefig(path)


    def bar_chart_bot(self, bot=10, path='bar_chart.png'):
        assert bot <= len(self.class_name)

        y_pos = range(bot,0,-1)
        plt.figure()
        plt.barh(y_pos, self.class_count[-bot:], align='center', alpha=0.5)
        plt.yticks(y_pos, self.class_name[-bot:])
        plt.xlabel('class name')
        plt.title('Statistic of the bot {} class in the dataset'.format(bot))
        plt.savefig(path)

    def hist(self, bins=10, path='hist_chart.png'):
        plt.figure()
        plt.hist(self.class_count)
        plt.savefig(path)

class AccuracyStat:
    '''This class is for computing the accuracy by class'''
    def __init__(self, num_class):
        self.num_class = num_class
        self.initialize()

    def initialize(self):
        self.pred_total = [0 for i in range(self.num_class)]
        self.pred_correct = [0 for i in range(self.num_class)]

    def accumulate(self, pred, y):
        for i in range(len(pred)):
            label = y.data[i]
            self.pred_total[label] += 1
            if label == pred.data[i]:
                self.pred_correct[label] += 1

    def compute_acc(self):
        self.acc_by_class = [self.pred_correct[i] / self.pred_total[i] if
                             self.pred_total[i] != 0 else 0 for i in range(self.num_class) ]

    def acc_demo(self, path='acc_demo.png'):
        self.compute_acc()
        plt.figure()
        y_pos = range(self.num_class, 0, -1)
        plt.barh(y_pos, self.acc_by_class, align='center', alpha=0.5)
        # plt.yticks(y_pos, self.class_name[-bot:])
        plt.xlabel('class name')
        plt.title('Statistic of the accuracy by class in the dataset')
        plt.savefig(path)



if __name__ == '__main__':
    # root_path = '/g/data/ll21/Kinetics700/validate'
    # stat = DatasetStatistics(root_path)
    # stat.bar_chart_top(top=20, path='./bar_chart')
    # stat.hist(path='./hist_chart.png')
    # print(stat.total)
    import torch
    acc_stat = AccuracyStat(10)

    pred = torch.tensor([i for i in range(10)])
    y = torch.tensor([i for i in range(10)])
    y[-1] = 0
    pred[2] = 11
    print(pred)
    print(y)
    acc_stat.accumulate(pred, y)
    acc_stat.acc_demo()




