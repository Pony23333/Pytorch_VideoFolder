# VideoLoader for loading raw videos using cv2
# The input videos can in different length or have different frame size.
# The VideoLoader can sample the videos and handle video with different size on-the-fly
# Jiteng Ma <majiteng123@outlook.com>

import torch.utils.data as data
import os
import os.path
import numpy as np
import cv2
import torch
import random


VIDEO_EXTENSIONS = ('.mp4','avi','rmvb')


def is_video_file(filename):
    return any(filename.endswith(extension) for extension in VIDEO_EXTENSIONS)


def find_classes(dir):
    ''' Make a mapping from class name to index'''
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx







def video_loader(path, size, model='2d',length=10):
    '''
    This function is a novel for video loading. The input video can be in any size
    and any length and equally sample the video with total frames = length. This function
    will convert the input video on the fly and return a np.array of fixed size.

    :param path: the absolute path of the video file
    :param size: the expected size of the output in tuple(w, h) or int
    :param model: 2d -> return TxCxHxW. 3d -> return CxTxHxW
    :param length: the number of frames extract
    :return: a np.array whose shape depends on model
    '''

    cap = cv2.VideoCapture(path)
    h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    num_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    if not isinstance(size,tuple):
        size = (size,size)

    frames = np.zeros((length, size[0], size[1], 3))
    rate = int(num_frames/length)
    for f in range(length):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f*rate)
        ret, frame = cap.read()
        # if the frame is read successfully
        if ret:
            # center crop the frame
            i = int(round((h - size[0]) / 2.))
            j = int(round((w - size[1]) / 2.))
            frame = frame[i:-i, j:-j, :]
            frames[f, :, :, :] = frame


        else:
            print("Video {} is Skipped!".format(path))
            break

    frames = frames.astype('float32')

    # 2d -> return TxCxHxW
    if model=='2d':
        return frames.transpose([0, 3, 1, 2])
    # 3d -> return CxTxHxW
    return frames.transpose([3, 0, 1, 2])




class VideoFolder(data.Dataset):
    '''
    A class of Dataset.
    :param root: the root folder of the dataset
    :param model: 2d -> return TxCxHxW. 3d -> return CxTxHxW
    :param transform: for data augmentation
    '''
    def __init__(self, root, model='2d', balanced=False, transform=None, target_transform=None,
                 loader=video_loader, size=112):
        classes, class_to_idx = find_classes(root)
        self.videos = self.make_dataset(root, class_to_idx) if not balanced else\
            self.make_balanced_dataset(root, class_to_idx)
        self.root = root
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.model = model
        self.size = size

    def make_dataset(self, dir, class_to_idx):
        ''' This function is for making dataset when data is saved under directory with correct naming.
        The return value is a tuple: (class, file_path)'''

        videos = []
        dir = os.path.expanduser(dir)  # expand the user's home directory to a absolute path
        for target in sorted(os.listdir(dir)):
            d = os.path.join(dir, target)
            if not os.path.isdir(d):
                continue

            for root, _, fnames in sorted(os.walk(d)):
                for fname in fnames:
                    if is_video_file(fname):
                        path = os.path.join(root, fname)
                        item = (path, class_to_idx[target])
                        videos.append(item)

        return videos

    def make_balanced_dataset(self, dir, class_to_idx, p=1.2):
        videos = []
        self.dataset_count = {}
        dir = os.path.expanduser(dir)  # expand the user's home directory to a absolute path
        for target in sorted(os.listdir(dir)):
            d = os.path.join(dir, target)
            if not os.path.isdir(d):
                continue
            # check the size of the folder, update the min_size
            length = len([name for name in os.listdir(d) if is_video_file(name)])
            self.dataset_count[target] = length

        # after we have the min_size of all the categories
        min_size = min(self.dataset_count.values())
        threshold = int(1.2 * min_size)

        for target in sorted(os.listdir(dir)):
            d = os.path.join(dir, target)
            if not os.path.isdir(d):
                continue

            fnames = os.listdir(d)
            if self.dataset_count[target] > threshold:
                fnames = random.sample(fnames, threshold)

            for fname in fnames:
                if is_video_file(fname):
                    path = os.path.join(d, fname)
                    item = (path, class_to_idx[target])
                    videos.append(item)

        return videos


    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path, target = self.videos[index]

        video_frames = self.loader(path, self.size, self.model)
        if self.transform is not None:
            video_frames = self.transform(video_frames)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return video_frames, target

    def __len__(self):
        return len(self.videos)



# a simple test
if __name__ == '__main__':
    import time
    root_path = 'E:\\hmdb51_org'
    full_dataset = VideoFolder(root_path, model='2d', balanced=True)
    batch_size = 16

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    dataset_tr, dataset_val = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    dl = torch.utils.data.DataLoader(dataset_tr, batch_size=batch_size, shuffle=True, num_workers=5, pin_memory=True)
    vdl = torch.utils.data.DataLoader(dataset_val, batch_size=batch_size, shuffle=True, num_workers=5, pin_memory=True)

    dataloader = {'train': dl, 'val': vdl}

    count = 0
    start_time = time.time()
    for i in dl:
        print('loading the {}th video'.format(count))
        count+=1
        if count>= 9:
            end_time = time.time()
            break
    print('time cost', end_time - start_time, 's')