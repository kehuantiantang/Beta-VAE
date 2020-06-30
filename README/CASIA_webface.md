## Train β-VAE on CASIA_webface dataset

### 1. Configuration

Refer to the [config](../config/casia.yaml) file, and [model](../models/casia_model.py). The architecture seems like below:

| Encoder                                     | Decoder                                        |
| ------------------------------------------- | ---------------------------------------------- |
| Input, 64x64x3                              | Input, 32                                      |
| Conv 32x4x4, stride 2, **RELU**, padding 1  | FC, 256, , **RELU**                            |
| Conv 32x4x4, stride 2, **RELU**, padding 1  | Upconv, 256x4x4, , **RELU**                    |
| Conv 64x4x4, stride 2, **RELU**, padding 1  | Upconv, 64x4x4, stride 2, **RELU**, padding 1  |
| Conv 64x4x4, stride 2, **RELU**, padding 1  | Upconv, 64x4x4, stride 2,  **RELU**, padding 1 |
| Conv 256x4x4, stride 1, **RELU**, padding 1 | Upconv, 32x4x4, stride 2, **RELU**, padding 1  |
| FC, 256, 2*32 (latent)                      | Upconv, 32x4x4, stride 2, padding 1            |

*Beside, I test [AlexNet](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf) and [ResNet](https://arxiv.org/abs/1512.03385) backbone as encoder part, you can refer the source code to get more information.*

### 2. Training Curve

#### 2.1 VAE
<table align='center'>
<tr align='center'>
<th> VAE Loss</th>
<th> β-VAE (β = 20)</th>
</tr>
<tr align='left'>
<td><img src = './res/celeba/vae_loss.png' width="600">
<td><img src = "./res/celeba/vae20_loss.png" width="600">
</tr>
</table>


### 3. Experiment Result 

#### 3.1 Feret

<table align='center'>
<tr align='center'>
<th> VAE fix noise</th>
<th> β-VAE (β = 20) fix noise</th>
</tr>
<tr>
<td><img src = 'res/celeba/vae_fix.png' height='400'>
<td><img src = 'res/celeba/vae20_fix.png'height='400'>
</tr>
<tr align='center'>
<th> VAE Random</th>
<th> β-VAE (β = 20) Random</th>
</tr>
<tr>
<td><img src = 'res/celeba/vae_random.png' height='400'>
<td><img src = 'res/celeba/vae20_random.png' height='400'>
</tr>
</table>

<table align='center'>
<tr align='center'>
  <th> Other architecture</th>
</tr>
<tr align='center'>
<td><img src = 'res/casia/res_random.png'>
</tr>
</table>

<table align='center'>
<tr align='center'>
  <th> InfoGAN </th>
</tr>
<tr align='center'>
<td><img src = 'res/celeba/c.png'>
</tr>
</table>