# $\beta$-VAE PyTorch

PyTorch implementation of [beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework](https://openreview.net/forum?id=Sy2fzU9gl) with result of experiments on *Faces*, *CelebA* and *CASIA-Webface*, *synthetic 2D shares* datasets.

## Deployment Environment
* Ubuntu 16.04 LTS

* NVIDIA TITAN XP

* cuda 10.1

* Python 3.8.3,, recommend to install package using **conda**

  * `conda env create -f beta.yml`

  ```yaml
  name: beta
  channels:
    - pytorch
    - defaults
  dependencies:
    - _libgcc_mutex=0.1=main
    - blas=1.0=mkl
    - ca-certificates=2020.1.1=0
    - certifi=2020.6.20=py38_0
    - cudatoolkit=10.1.243=h6bb024c_0
    - freetype=2.10.2=h5ab3b9f_0
    - intel-openmp=2020.1=217
    - jpeg=9b=h024ee3a_2
    - ld_impl_linux-64=2.33.1=h53a641e_7
    - libedit=3.1.20191231=h7b6447c_0
    - libffi=3.3=he6710b0_1
    - libgcc-ng=9.1.0=hdf63c60_0
    - libgfortran-ng=7.3.0=hdf63c60_0
    - libpng=1.6.37=hbc83047_0
    - libstdcxx-ng=9.1.0=hdf63c60_0
    - libtiff=4.1.0=h2733197_1
    - lz4-c=1.9.2=he6710b0_0
    - mkl=2020.1=217
    - mkl-service=2.3.0=py38he904b0f_0
    - mkl_fft=1.1.0=py38h23d657b_0
    - mkl_random=1.1.1=py38h0573a6f_0
    - ncurses=6.2=he6710b0_1
    - ninja=1.9.0=py38hfd86e86_0
    - numpy=1.18.5=py38ha1c710e_0
    - numpy-base=1.18.5=py38hde5b4d6_0
    - olefile=0.46=py_0
    - openssl=1.1.1g=h7b6447c_0
    - pillow=7.1.2=py38hb39fc2d_0
    - pip=20.1.1=py38_1
    - python=3.8.3=hcff3b4d_0
    - pytorch=1.5.1=py3.8_cuda10.1.243_cudnn7.6.3_0
    - readline=8.0=h7b6447c_0
    - setuptools=47.3.1=py38_0
    - six=1.15.0=py_0
    - sqlite=3.32.3=h62c20be_0
    - tk=8.6.10=hbc83047_0
    - torchvision=0.6.1=py38_cu101
    - wheel=0.34.2=py38_0
    - xz=5.2.5=h7b6447c_0
    - zlib=1.2.11=h7b6447c_3
    - zstd=1.4.4=h0b5b093_3
    - pip:
      - absl-py==0.9.0
      - argparse==1.4.0
      - cachetools==4.1.0
      - chardet==3.0.4
      - easydict==1.9
      - google-auth==1.18.0
      - google-auth-oauthlib==0.4.1
      - grpcio==1.30.0
      - idna==2.9
      - markdown==3.2.2
      - oauthlib==3.1.0
      - protobuf==3.12.2
      - pyasn1==0.4.8
      - pyasn1-modules==0.2.8
      - pynvml==8.0.4
      - pyyaml==5.3.1
      - requests==2.24.0
      - requests-oauthlib==1.3.0
      - rsa==4.6
      - tensorboard==2.2.2
      - tensorboard-plugin-wit==1.6.0.post3
      - tqdm==4.46.1
      - urllib3==1.25.9
      - werkzeug==1.0.1
  prefix: /home/{username}/anaconda3/envs/beta
  ```

  

Edit the *argparse* in **`main.py`** file to select training parameters and the dataset to use.

```python
parser = argparse.ArgumentParser(description='Beta-VAE')

parser.add_argument('--train', default=True, help='train or traverse')
# repeatable
parser.add_argument('--seed', default=1, type=int, help='random seed')
parser.add_argument('--max_iter', default=1e6, type=int, help='maximum training iteration')

parser.add_argument('--viz_on', default=True, help='enable tensorboard visualization')
#flush to tensorboard or not
parser.add_argument('--save_output', default=True, help='save traverse images and gif')

parser.add_argument('--gather_step', default=2000, type=int, help='numer of iterations after which data is gathered')
# Visualization
parser.add_argument('--display_step', default=2000, type=int, help='number of iterations after which loss data is printed and vision is updated')
parser.add_argument('--save_step', default=4000, type=int, help='number of iterations after which a checkpoint is saved')

# paraterment for beta
parser.add_argument('--beta', dest='beta', default=250, type=float)
# inference
parser.add_argument('--checkpoint_file', dest='checkpoint_file', default=None, type=str)
# training dataset
parser.add_argument('--dataset', dest='dataset', help='Training dataset', default='casia', type=str)

# output folder
parser.add_argument('--output_dir', dest='output_dir', help='the dir save result', default='output1', type=str)
parser.add_argument('--comment', type=str, default='')
args = parser.parse_args()
```

Also, you can modify the {dataset}.yaml file in config to specify the training hyperparameters. This is a example of training hyperparamters, all the {dataset}.yaml will inherit the base.yaml file:

```yaml
# base.yaml
lr: 0.0001
beta1: 0.9
beta2: 0.999
batch_size: 64
epochs: 500
max_iter: 1e6

# specify
z_dim: 32
distribution: bernoulli
nb_channels: 3
image_size: 64
```
Example of training the faces dataset, the config file will find corresponding model and config file when specify the dataset parameter.

```sh
python3 main.py --dataset mnist --output_folder output --beta 250
```



## Model

My code follow the Table 1 to implement the dataset *CelebA* and *Faces*. I create a [base class.py](../models/vae_base.py) and implement the fundamental    method for training VAE, you can inherit this class and only write encoder, decoder part for your specific dataset.  For example:

```python
class BetaVAE(VAEBase):
    '''Inhert the VAEBase class'''

    def __init__(self, z_dim=10, nb_channel=1):
        super(BetaVAE, self).__init__()
        self.nb_channel = nb_channel
        self.z_dim = z_dim

        self.encoder = ...
        self.decoder = ...
        self.weight_init()
```



## Results

* **Faces** - [`faces.md`](./README/faces.md)

* **CelebA** - [`CelebA.md`](./README/CelebA.md)

* **CASIA_webface** - [`CASIA_webface.md`](./README/CASIA_webface.md)

* **2D shapes** -[`2d_shapes.md`](./README/2d_shapes.md)

* **Info-GAN, VAE, $\beta$-VAE comparsion**

  

