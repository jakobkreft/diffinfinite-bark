import torch
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
from PIL import Image
import numpy as np
import torchvision.transforms as T
from dm import Unet, GaussianDiffusion, Trainer
from random_diffusion import save_tensor_as_png

device = torch.device('cuda:0')  # cuda:0 because CUDA_VISIBLE_DEVICES remaps GPU 1 to index 0

# Load model
unet = Unet(dim=256, num_classes=4, dim_mults=[1,2,4], channels=4,
          resnet_block_groups=2, block_per_layer=2)
model = GaussianDiffusion(unet, image_size=64, timesteps=1000,
                        sampling_timesteps=250, loss_type='l2')
trainer = Trainer(model, train_batch_size=2, train_lr=1e-4, train_num_steps=200000,
                save_and_sample_every=99999, num_workers=0,
                results_folder='./results/bark_200k')
trainer.load(5)  # <-- replace with your milestone number

# Load a mask
mask = np.array(Image.open('D:/jk/diffinfinite_512/bark_0100_mask.png'))
mask_tensor = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).float().to(device)

# Generate
trainer.ema.to(device)
trainer.ema = trainer.ema.eval()
vae = trainer._get_vae().to(device)
z = torch.ones((1, 4, 64, 64), device=device)
with torch.no_grad():
  z = trainer.ema.sample(z, mask_tensor, cond_scale=3.0) * 50
  img = torch.clip(vae.decode(z).sample, 0, 1)
save_tensor_as_png(img[0], 'generated_sample.png')
print('Saved to generated_sample.png')