# coding=utf-8
import subprocess

import torch
from torchvision import transforms
import os.path as osp
import torchvision.datasets as dsets

root = 'data/'
# root = ''

def create_softlink():
    # TODO create datset softlink from infogan to beta_vae
    pass


def get_data(dataset, batch_size):

    # Get MNIST dataset.
    if dataset == 'MNIST':
        transform = transforms.Compose([
            transforms.Resize(28),
            transforms.CenterCrop(28),
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
            transforms.Resize((64,64)),
            transforms.ToTensor()])

        celeba_path = osp.join(root, 'celeba')
        if not osp.exists(celeba_path):
            print("Start to run download script")
            subprocess.call('./celeba.sh', shell=True)
            print("Dataset download finished!")
        dataset = dsets.ImageFolder(root=celeba_path, transform=transform)

    elif dataset == 'casia_webface' or dataset.lower() == 'casia':
        transform = transforms.Compose([
            transforms.Resize(68),
            transforms.CenterCrop(65),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5),
                                 (0.5, 0.5, 0.5))])

        casia_path = osp.join(root, 'casia_webface')
        dataset = dsets.ImageFolder(root=casia_path, transform=transform)

    elif dataset == 'Faces' or dataset.lower() == 'faces':
        face_path = osp.join(root, 'faces')
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize(40),
            transforms.CenterCrop(32),
            transforms.ToTensor()])
        dataset = dsets.ImageFolder(root=face_path, transform=transform)
    else:
        raise ValueError('Dataset name %s is wrong'%dataset)

    # Create dataloader.
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=True, num_workers=2, pin_memory = True)

    return dataloader

if __name__ == '__main__':
    dataloader = get_data('celeba', 32)
    import time
    start = time.time()
    for i, (x, _) in enumerate(dataloader):
        print(x.size())
        if i == 50:
            break
    print(time.time() -start)


