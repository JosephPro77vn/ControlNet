import argparse
import torch
import yaml
from scheduler import LinearScheduler
from sd.datasets.celeb_dataset import CelebDataset
from torch.utils.data import DataLoader
import os
import model_loader
from controlnet import ControlNet
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm
import numpy as np
from transformers import CLIPTokenizer
from utils.config_utils import *
from sd.datasets.vehicle_dataset import VehicleDataset

WIDTH = 512
HEIGHT = 512
LATENTS_WIDTH = WIDTH // 8
LATENTS_HEIGHT = HEIGHT // 8

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.backends.mps.is_available():
    device = torch.device('mps')
    print('Using mps')

# device = 'cpu'


def train(args):
    # Read the config file #
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    print(config)

    diffusion_config = config['diffusion_params']
    dataset_config = config['dataset_params']
    diffusion_model_config = config['ldm_params']
    autoencoder_model_config = config['autoencoder_params']
    train_config = config['train_params']


    model_file = "D:\python\ptyhonProject\pythonfile\Letstry\pythonProject\StableDiffusion\StableDiffusion\model\weights\sd-v1-4.ckpt"

    models = model_loader.preload_models_from_standard_weights(model_file, device)
    #clip = CLIP()
    # state_dict = model_converter.load_from_standard_weights(model_file, device)
    # clip_state_dict = model_loader.preload_models_from_standard_weights(ckpt_path=model_file, device=device)
    print("loading clip CheckPoint....")
    clip = models['clip']



    scheduler = LinearScheduler(num_timesteps=diffusion_config['num_timesteps'],
                                     beta_start=diffusion_config['beta_start'],
                                     beta_end=diffusion_config['beta_end'])
    condition_types = []
    condition_config = get_config_value(diffusion_model_config, key='condition_config', default_value=None)
    if condition_config is not None:
        assert 'condition_types' in condition_config, \
            "condition type missing in conditioning config"
        condition_types = condition_config['condition_types']
        if 'text' in condition_types:
            validate_text_config(condition_config)
            with torch.no_grad():
                tokenizer = CLIPTokenizer("../data/tokenizer_vocab.json", merges_file="../data/tokenizer_merges.txt")
        elif 'class' in condition_types:
            validate_text_config(condition_config)
            with torch.no_grad():
                tokenizer = CLIPTokenizer("../data/tokenizer_vocab.json", merges_file="../data/tokenizer_merges.txt")
    im_dataset_cls = {
        # 'mnist': MnistDataset,
        'celebhq': CelebDataset,
        'vehicle': VehicleDataset
    }.get(dataset_config['name'])
    im_dataset = im_dataset_cls(split='train',
                              im_path=dataset_config['im_path'],
                              im_size=dataset_config['im_size'],
                              im_channels=dataset_config['im_channels'],
                              use_latents=False,
                              latent_path=None,
                              return_hint=True,
                              condition_config=condition_config)

    data_loader = DataLoader(im_dataset,
                             batch_size=train_config['ldm_batch_size'],
                             shuffle=True)

    dataiter = iter(data_loader)
    images, hint, input = next(dataiter)
    print(images.shape)
    print(hint.shape)
    print(input)
    # Instantiate the model
    # downscale factor = canny_image_size // latent_size
    # latent_size = dataset_config['im_size'] // 2 ** sum(autoencoder_model_config['down_sample'])
    # downscale_factor = dataset_config['canny_im_size'] // latent_size
    downscale_factor = 32

    model = ControlNet(im_channels=autoencoder_model_config['z_channels'],
                       model_config=diffusion_model_config,
                       model_locked=True,
                       model_ckpt=models,
                       device=device,
                       down_sample_factor=downscale_factor).to(device)
    model.train()

    # Create output directories
    if not os.path.exists(train_config['task_name']):
        os.mkdir(train_config['task_name'])

    #     # Load checkpoint if found
    # if os.path.exists(os.path.join(train_config['task_name'], train_config['controlnet_ckpt_name'])):
    #     print('Loading checkpoint as found one')
    #     model.load_state_dict(torch.load(os.path.join(train_config['task_name'],
    #                                                   train_config['controlnet_ckpt_name']),
    #                                      map_location=device))

    # encoder = VAE_Encoder()
    # state_dict = model_loader.preload_models_from_standard_weights(ckpt_path=model_file, device=device)
    print("loading encoder checkpoint.....")
    # encoder.load_state_dict(state_dict['encoder'], strict=True)

    encoder = models['encoder']

    num_epochs = train_config['ldm_epochs']
    optimizer = Adam(model.get_params(), lr=train_config['controlnet_lr'])
    lr_scheduler = MultiStepLR(optimizer, milestones=train_config['controlnet_lr_steps'], gamma=0.1)
    criterion = torch.nn.MSELoss()
    step_count = 0
    count = 0
    latents_shape = (1, 4, LATENTS_HEIGHT, LATENTS_WIDTH)

    for epoch_idx in range(num_epochs):
        losses = []
        cond_input = None
        for data in tqdm(data_loader):
            # print(data)
            if condition_config is not None:
                im, hint, cond_input = data
                assert not torch.isnan(im).any(), "NaN found in images!"
                assert not torch.isinf(im).any(), "Inf found in images!"
                assert not torch.isinf(hint).any(), "Inf hint found in hint"
                assert not torch.isnan(hint).any(), "NaN hint found in hint"

            else:
                im, hint = data
            optimizer.zero_grad()
            im = im.float().to(device)

            with torch.no_grad():
                generator = torch.Generator(device=device)
                encoder_noise = torch.randn(latents_shape, generator=generator, device=device)
                im = encoder(im, encoder_noise)
            if 'text' in condition_types:
                with torch.no_grad():
                    assert 'text' in cond_input, 'Conditioning Type Text but no text conditioning input present'
                    validate_text_config(condition_config)
                    prompt = cond_input['text']
            elif 'class' in condition_types:
                with torch.no_grad():
                    assert 'class' in cond_input, 'Conditioning Type Text but no text conditioning input present'
                    validate_text_config(condition_config)
                    prompt = cond_input['class']


            hint = hint.float().to(device)
            # Sample random noise
            noise = torch.randn_like(im).to(device)

            # Sample timestep
            t = torch.randint(0, diffusion_config['num_timesteps'], (im.shape[0],)).to(device)
            print(t)
            tokens = tokenizer.batch_encode_plus(
                prompt, padding="max_length", max_length=77   # should have been text in empty String if there is text input
            ).input_ids
            # (Batch_Size, Seq_Len)
            tokens = torch.tensor(tokens, dtype=torch.long, device=device)
            # (Batch_Size, Seq_Len) -> (Batch_Size, Seq_Len, Dim)
            context = clip(tokens)
            # print('clip context', context)

            # Add noise to images according to timestep
            noisy_im = scheduler.add_noise(im, noise, t)

            optimizer.step()
            # torch.autograd.set_detect_anomaly(True)

            noise_pred = model(noisy_im, context, hint, t)
            assert not torch.isnan(noise).any(), "NaN in target!"
            # assert not torch.isnan(noise_pred).any(), "NaN in model output!"
            assert not torch.isinf(noise_pred).any(), "Inf in model output!"
            assert not torch.isinf(noise).any(), "Inf in target!"
            loss = criterion(noise_pred, noise)
            loss.backward()

            for name, param in model.named_parameters():
                if param.grad is not None:
                    if torch.isnan(param.grad).any():
                        print(f"NaN gradient in {name}!")
            losses.append(loss.item())
            step_count += 1
        print('Finished epoch:{} | Loss : {:.4f}'.format(
            epoch_idx + 1,
            np.mean(losses)))
        lr_scheduler.step()
        torch.save(model.state_dict(), os.path.join(train_config['task_name'],
                                                    train_config['controlnet_ckpt_name']))

    print('Done Training ...')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for ldm controlnet training')
    parser.add_argument('--config', dest='config_path',
                        default='../config/vehicle.yaml', type=str)
    args = parser.parse_args()
    train(args)