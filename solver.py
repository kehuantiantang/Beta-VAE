"""solver.py"""

import warnings
from torch.utils.tensorboard import SummaryWriter

from dataset import get_data

warnings.filterwarnings("ignore")

import os
import os.path as osp
from tqdm import trange

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.utils import make_grid, save_image

from utils import grid2gif

device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

def reconstruction_loss(x, x_recon, distribution):
    batch_size = x.size(0)
    assert batch_size != 0

    if distribution == 'bernoulli':
        recon_loss = F.binary_cross_entropy_with_logits(x_recon, x, size_average=False).div(batch_size)
    elif distribution == 'gaussian':
        x_recon = F.sigmoid(x_recon)
        recon_loss = F.mse_loss(x_recon, x, size_average=False).div(batch_size)
    else:
        recon_loss = None

    return recon_loss


def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5*(1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld


class DataGather(object):
    def __init__(self):
        self.data = self.get_empty_data_dict()

    def get_empty_data_dict(self):
        return dict(iter=[],
                    recon_loss=[],
                    total_kld=[],
                    dim_wise_kld=[],
                    mean_kld=[],
                    mu=[],
                    var=[],
                    images=[],)

    def insert(self, **kwargs):
        for key in kwargs:
            self.data[key].append(kwargs[key])

    def flush(self):
        self.data = self.get_empty_data_dict()


class Solver(object):
    def __init__(self, params):
        self.prepare_output_folder(params)

        self.global_iter = 0

        self.max_iter = params.max_iter
        self.params = params
        self.beta = params.beta

        self.net = self.get_model(params)
        self.viz_on = params.viz_on

        if self.viz_on:
            self.writer = SummaryWriter(params.summary)

        
        self.load_checkpoint(params.checkpoint_file)

        self.save_output = params.save_output

        self.gather_step = params.gather_step
        self.display_step = params.display_step
        self.save_step = params.save_step

        self.dataset = params.dataset
        self.batch_size = params.batch_size
        self.data_loader = get_data(params.dataset, params.batch_size)

        self.gather = DataGather()

        self.set_optimizer(params)

    
    def prepare_output_folder(self, params):
        output_folder = params.output_dir  # 'output1'
        params['summary'] = osp.join('./', output_folder, 'summary', params['dataset'] + params.comment)
        params['checkpoint'] = osp.join('./', output_folder, 'checkpoint', params['dataset'] + params.comment)
        params['info'] = osp.join('./', output_folder, 'info', params['dataset'] + params.comment)
        os.makedirs(params['summary'], exist_ok=True)
        os.makedirs(params['checkpoint'], exist_ok=True)
        os.makedirs(params['info'], exist_ok=True)

        print('Save summary to %s '%params['summary'])
    
    def set_optimizer(self, params):
        self.optim = optim.Adam(self.net.parameters(), lr=params.lr,
                                betas=(params.beta1, params.beta2))

    def get_model(self, params):

        if params.dataset == 'celeba':
            from model.celea_model import BetaVAE
        elif params.dataset == 'casia':
            from model.casia_model import BetaVAE
        elif params.dataset == 'faces':
            from model.faces_model import BetaVAE
        elif params.dataset == '2dshapes':
            from model.twoD_model import BetaVAE
            self.C_max = Variable((torch.FloatTensor([self.params.C_max]))).to(device)
        else:
            raise NotImplementedError('Get model %s'%params.dataset)

        return BetaVAE(params.z_dim, params.nb_channels).to(device)

    def get_loss(self, recon_loss, total_kld, **kwargs):

        if self.params.dataset == '2dshapes':
            input = self.params.C_max/self.params.C_stop_iter* kwargs.get('current_iter')
            C = torch.clamp(torch.FloatTensor([input]), 0, self.C_max.data[0]).cuda()
            if self.params.beta != 1:
                beta_vae_loss = recon_loss + self.params.gamma*(total_kld-C).abs()
            else:
                beta_vae_loss = recon_loss + total_kld
        else:
            beta_vae_loss = recon_loss + self.beta*total_kld
        return beta_vae_loss

    def train(self):
        self.net_mode(train=True)

        pbar = trange(self.global_iter, int(self.max_iter))
        for epoch in range(self.params.epochs):

            for iter, (x, _) in enumerate(self.data_loader):
                current_iter = epoch * len(self.data_loader) + iter

                x = Variable(x).to(device)
                x_recon, mu, logvar = self.net(x)
                recon_loss = reconstruction_loss(x, x_recon, self.params.distribution)
                total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)

                beta_vae_loss = self.get_loss(recon_loss, total_kld, current_iter= current_iter)

                self.optim.zero_grad()
                beta_vae_loss.backward()
                self.optim.step()

                if self.viz_on and current_iter%self.gather_step == 0:
                    self.gather.insert(iter=current_iter,
                                       mu=mu.mean(0).data, var=logvar.exp().mean(0).data,
                                       recon_loss=recon_loss, total_kld=total_kld.data,
                                       dim_wise_kld=dim_wise_kld.data, mean_kld=mean_kld.data)

                if current_iter%self.display_step == 0:
                    pbar.write('[{}] recon_loss:{:.3f} total_kld:{:.3f} mean_kld:{:.3f}'.format(
                        current_iter, recon_loss.item(), total_kld.data[0], mean_kld.data[0]))

                    var = logvar.exp().mean(0).data
                    var_str = ''
                    for j, var_j in enumerate(var):
                        var_str += 'var{}:{:.4f} '.format(j+1, var_j)
                    pbar.write(var_str)

                    # if self.objective == 'B':
                    #     pbar.write('C:{:.3f}'.format(C.data[0]))

                    if self.viz_on:
                        self.gather.insert(images=x.data)
                        self.gather.insert(images=F.sigmoid(x_recon).data)
                        self.viz_reconstruction(current_iter)
                        self.viz_lines(current_iter)
                        self.gather.flush()

                    if self.viz_on or self.save_output:
                        self.viz_traverse(current_iter)

                if current_iter%self.save_step == 0:
                    self.save_checkpoint(self.params, 'last')
                    pbar.write('Saved checkpoint(iter:{})'.format(current_iter))

                if current_iter%50000 == 0:
                    self.save_checkpoint(self.params, '%08d'%current_iter)

                pbar.update()

            if pbar.n >= int(self.max_iter):
                break

        pbar.write("[Training Finished]")
        pbar.close()

    def viz_reconstruction(self, current_iter):
        self.net_mode(train=False)
        x = self.gather.data['images'][0][:100]
        x = make_grid(x, normalize=True)
        x_recon = self.gather.data['images'][1][:100]
        x_recon = make_grid(x_recon, normalize=True)
        images = torch.stack([x, x_recon], dim=0).cpu()

        for i in range(images.size()[0]):
            self.writer.add_image('recon/%s'%i, images[i], current_iter)

        self.net_mode(train=True)

    def viz_lines(self, current_iter):
        self.net_mode(train=False)
        recon_losses = torch.stack(self.gather.data['recon_loss']).cpu()

        mus = torch.stack(self.gather.data['mu']).cpu()
        vars = torch.stack(self.gather.data['var']).cpu()

        dim_wise_klds = torch.stack(self.gather.data['dim_wise_kld'])
        mean_klds = torch.stack(self.gather.data['mean_kld'])
        total_klds = torch.stack(self.gather.data['total_kld'])
        klds = torch.cat([dim_wise_klds, mean_klds, total_klds], 1).cpu()
        iters = torch.Tensor(self.gather.data['iter'])

        legend = []
        for z_j in range(self.params.z_dim):
            legend.append('z_{}'.format(z_j))

        self.writer.add_scalar('Recon_loss', recon_losses, current_iter)
        self.writer.add_scalar('kl_divergence', total_klds, current_iter)
        self.writer.add_scalars('posterior_mean', dict(zip(legend, mus.view(self.params.z_dim)[:5])), current_iter)
        self.writer.add_scalars('posterior_vars', dict(zip(legend, vars.view(self.params.z_dim)[:5])), current_iter)
        self.net_mode(train=True)

    def viz_traverse(self, current_iter, limit=3, inter=2/3, loc=-1):
        self.net_mode(train=False)
        import random

        decoder = self.net.decoder
        encoder = self.net.encoder
        interpolation = torch.arange(-limit, limit+0.1, inter)

        n_dsets = len(self.data_loader.dataset)
        rand_idx = random.randint(1, n_dsets-1)

        random_img, _ = self.data_loader.dataset.__getitem__(rand_idx)
        random_img = Variable(random_img, volatile=True).unsqueeze(0).to(device)
        random_img_z = encoder(random_img)[:, :self.params.z_dim]

        random_z = Variable(torch.rand(1, self.params.z_dim), volatile=True).to(device)

        if self.dataset == 'dsprites':
            fixed_idx1 = 87040 # square
            fixed_idx2 = 332800 # ellipse
            fixed_idx3 = 578560 # heart

            fixed_img1 = self.data_loader.dataset.__getitem__(fixed_idx1)
            fixed_img1 = Variable(fixed_img1, volatile=True).unsqueeze(0).to(device)
            fixed_img_z1 = encoder(fixed_img1)[:, :self.params.z_dim]

            fixed_img2 = self.data_loader.dataset.__getitem__(fixed_idx2)
            fixed_img2 = Variable(fixed_img2, volatile=True).unsqueeze(0).to(device)
            fixed_img_z2 = encoder(fixed_img2)[:, :self.params.z_dim]

            fixed_img3 = self.data_loader.dataset.__getitem__(fixed_idx3)
            fixed_img3 = Variable(fixed_img3, volatile=True).unsqueeze(0).to(device)
            fixed_img_z3 = encoder(fixed_img3)[:, :self.params.z_dim]

            Z = {'fixed_square':fixed_img_z1, 'fixed_ellipse':fixed_img_z2,
                 'fixed_heart':fixed_img_z3, 'random_img':random_img_z}
        else:
            fixed_idx = 0
            fixed_img, _ = self.data_loader.dataset.__getitem__(fixed_idx)
            fixed_img = Variable(fixed_img, volatile=True).unsqueeze(0).to(device)
            fixed_img_z = encoder(fixed_img)[:, :self.params.z_dim]

            Z = {'fixed_img':fixed_img_z, 'random_img':random_img_z, 'random_z':random_z}

        gifs = []

        for key in list(Z.keys()):
            z_ori = Z[key]
            samples = []
            for row in range(self.params.z_dim):
                if loc != -1 and row != loc:
                    continue
                z = z_ori.clone()
                for val in interpolation:
                    z[:, row] = val
                    sample = F.sigmoid(decoder(z)).data
                    samples.append(sample)
                    gifs.append(sample)
            samples = torch.cat(samples[:100], dim=0).cpu()

            if self.viz_on:
                images = make_grid(samples, nrow=10, padding=2, normalize=True)
                self.writer.add_image('Traverse/%s'%key, images, current_iter)


        if self.save_output:
            output_dir = self.params['info']
            gifs = torch.cat(gifs)
            gifs = gifs.view(len(Z), self.params.z_dim, len(interpolation), self.params.nb_channels, self.params.image_size, self.params.image_size).transpose(1, 2)
            for i, key in enumerate(Z.keys()):
                for j, val in enumerate(interpolation):
                    save_image(tensor=gifs[i][j].cpu(),
                               fp=os.path.join(output_dir, '{}_{}.jpg'.format(key, j)),
                               nrow=self.params.z_dim, pad_value=1)


                    # index = slice(len(interpolation) * i + j, len(interpolation) * i + j + self.params.z_dim)
                    # img = torch.cat(gifs[index][:100]).cpu()
                    # save_image(tensor=img,
                    #            fp=os.path.join(output_dir, '{}_{}.jpg'.format(key, j)),
                    #            nrow=self.params.z_dim, pad_value=1)


                grid2gif(os.path.join(output_dir, key+'*.jpg'),
                         os.path.join(output_dir, key+'.gif'), delay=10)

        self.net_mode(train=True)

    def net_mode(self, train):
        if not isinstance(train, bool):
            raise('Only bool type is supported. True or False')

        if train:
            self.net.train()
        else:
            self.net.eval()


    def save_checkpoint(self, params, filename, silent=True):
        model_states = {'net':self.net.state_dict(),}
        optim_states = {'optim':self.optim.state_dict(),}

        states = {'iter':self.global_iter,
                  'model_states':model_states,
                  'optim_states':optim_states}

        file_path = os.path.join(params.checkpoint, filename + '.pth')
        torch.save(states, file_path)
        if not silent:
            print("=> saved checkpoint '{}' (iter {})".format(file_path, self.global_iter))

    def load_checkpoint(self, file_path):
        if file_path is not None and os.path.isfile(file_path):
            checkpoint = torch.load(file_path)
            self.global_iter = checkpoint['iter']
            self.win_recon = checkpoint['win_states']['recon']
            self.win_kld = checkpoint['win_states']['kld']
            self.win_var = checkpoint['win_states']['var']
            self.win_mu = checkpoint['win_states']['mu']
            self.net.load_state_dict(checkpoint['model_states']['net'])
            self.optim.load_state_dict(checkpoint['optim_states']['optim'])
            print("=> loaded checkpoint '{} (iter {})'".format(file_path, self.global_iter))
        else:
            print("=> no checkpoint found at '{}'".format(file_path))
