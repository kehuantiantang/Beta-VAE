# coding=utf-8
import subprocess

import torch
from torch.utils.data import Dataset
from torchvision import transforms
import os.path as osp
import torchvision.datasets as dsets
import numpy as np


root = 'data/'
# root = ''


class CustomTensorDataset(Dataset):
    def __init__(self, data_tensor):
        self.data_tensor = data_tensor

    def __getitem__(self, index):
        return self.data_tensor[index], 0

    def __len__(self):
        return self.data_tensor.size(0)

def get_data(dataset, batch_size, image_size):

    # Get MNIST dataset.
    if dataset == 'MNIST':
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor()])

        dataset = dsets.MNIST(root+'mnist/', train='train',
                              download=True, transform=transform)

    # Get SVHN dataset.
    elif dataset == 'SVHN':
        transform = transforms.Compose([
            transforms.Resize(32),
            transforms.CenterCrop(32),
            transforms.ToTensor()])

        dataset = dsets.SVHN(root+'svhn/', split='train',
                             download=True, transform=transform)

    # Get FashionMNIST dataset.
    elif dataset == 'FashionMNIST':
        transform = transforms.Compose([
            transforms.Resize(28),
            transforms.CenterCrop(28),
            transforms.ToTensor()])

        dataset = dsets.FashionMNIST(root+'fashionmnist/', train='train',
                                     download=True, transform=transform)

    # Get CelebA dataset.
    # MUST ALREADY BE DOWNLOADED IN THE APPROPRIATE DIRECTOR DEFINED BY ROOT PATH!
    elif dataset == 'CelebA' or dataset.lower() == 'celeba':
        transform = transforms.Compose([
            transforms.Resize((image_size,image_size)),
            transforms.ToTensor()])

        celeba_path = osp.join(root, 'celeba')
        if not osp.exists(celeba_path):
            import subprocess
            print("Start to run download script")
            subprocess.call('./celeba.sh', shell=True)
            print("Dataset download finished!")
        dataset = dsets.ImageFolder(root=celeba_path, transform=transform)

    elif dataset == 'casia_webface' or dataset.lower() == 'casia':
        transform = transforms.Compose([transforms.Resize((image_size, image_size)), transforms.ToTensor()])

        casia_path = osp.join(root, 'casia_webface')
        dataset = dsets.ImageFolder(root=casia_path, transform=transform)

    elif dataset == 'Faces' or dataset.lower() == 'faces':
        # face_path = osp.join(root, 'feret')
        face_path = osp.join(root, 'faces')
        transform = transforms.Compose([
            transforms.Resize(64),
            transforms.ToTensor()])
        dataset = dsets.ImageFolder(root=face_path, transform=transform)
    elif dataset == '2dshapes':
        twi_dshapes_path = osp.join(root, 'dsprites-dataset', 'dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz')
        if not osp.exists(osp.dirname(twi_dshapes_path)):
            import subprocess
            print('Now download dsprites-dataset')
            subprocess.call(['./data/download_2dshape.sh'])
            print('Finished')
        data = np.load(twi_dshapes_path, encoding='bytes')
        data = torch.from_numpy(data['imgs']).unsqueeze(1).float()

        dataset = CustomTensorDataset(data_tensor=data)

    else:
        raise ValueError('Dataset name %s is wrong'%dataset)

    # Create dataloader.
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=True, num_workers=3, pin_memory = True)

    return dataloader

if __name__ == '__main__':
    dataloader = get_data('2dshapes', 32)
    import time
    start = time.time()
    for i, (x, _) in enumerate(dataloader):
        print(x.size())
        if i == 50:
            break
    print(time.time() -start)


