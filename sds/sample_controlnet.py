import torch
import torchvision
import argparse
import yaml
import os
import random
from torchvision.utils import make_grid
from tqdm import tqdm

from datasets.vehicle_dataset import VehicleDataset
from datasets.celeb_dataset import CelebDataset
from sds.scheduler import DDPMSampler
from sds.scheduler import get_time_embedding
from sds.controlnet import ControlNet
from transformers import CLIPTokenizer
from utils.config_utils import *
from sds.scheduler import LinearScheduler
from sds.model_loader import preload_models_from_standard_weights
WIDTH = 512
HEIGHT = 512
LATENTS_WIDTH = WIDTH // 8
LATENTS_HEIGHT = HEIGHT // 8

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.backends.mps.is_available():
    device = torch.device('mps')
    print('Using mps')

# device = 'cpu'



def sample(model, scheduler, train_config, diffusion_model_config, condition_types, tokenizer, condition_config,
           autoencoder_model_config, diffusion_config, dataset_config, encoder, decoder, clip, dataset):
    r"""
    Sample stepwise by going backward one timestep at a time.
    We save the x0 predictions
    """
    seed = train_config['seed']
    generator = torch.Generator(device=device)
    if seed is None:
        generator.seed()
    else:
        generator.manual_seed(seed)
    im_size= (1, 4, LATENTS_HEIGHT, LATENTS_WIDTH)
    xt = torch.randn(im_size, generator=generator, device=device)

    # Get random hints for the desired number of samples
    hints = []
    for idx in range(train_config['num_samples']):
        hint_idx = random.randint(0, len(dataset) - 1)  # Fix: Subtract 1 to avoid out-of-bounds
        _, hint, _ = dataset[hint_idx]  # Extract only the hint (ignore image and text)
        hints.append(hint.unsqueeze(0).to(device))  # Add batch dim and move to device

    # Concatenate all hints into a single tensor
    hints = torch.cat(hints, dim=0).to(device)

    # Output shape: [num_samples, C, H, W]
    print(hints.shape)

    # Save the hints
    hints_grid = make_grid(hints, nrow=train_config['num_grid_rows'])
    hints_img = torchvision.transforms.ToPILImage()(hints_grid)
    hints_img.save(os.path.join(train_config['task_name'], 'hint.png'))
    cond_input = {'text': ['']}
    if 'text' in condition_types:
        with torch.no_grad():
            # # assert 'text' in cond_input, 'Conditioning Type Text but no text conditioning input present'
            # validate_text_config(condition_config)
            prompt = cond_input
    elif 'class' in condition_types:
        with torch.no_grad():
            # assert 'class' in cond_input, 'Conditioning Type Text but no text conditioning input present'
            # validate_text_config(condition_config)
            prompt = cond_input

    tokens = tokenizer.batch_encode_plus(
        prompt, padding="max_length", max_length=77  # should have been text in empty String if there is text input
    ).input_ids
    # (Batch_Size, Seq_Len)
    tokens = torch.tensor(tokens, dtype=torch.long, device=device)
    # (Batch_Size, Seq_Len) -> (Batch_Size, Seq_Len, Dim)
    context = clip(tokens)
    n_inference_steps = 100
    sampler = DDPMSampler(generator)
    sampler.set_inference_timesteps(n_inference_steps)

    # timesteps = tqdm(sampler.timesteps)

    for i in tqdm(reversed(range(diffusion_config['num_timesteps']))):
    # for  i, timestep in enumerate(timesteps):

        # time_embedding = get_time_embedding(timestep).to(device)
        # Get prediction of noise
        noise_pred = model(xt, context, hints, torch.as_tensor(i).unsqueeze(0).to(device))

        # noise_pred = model(xt, context, hints, time_embedding.to(device))

        # Use scheduler to get x0 and xt-1
        xt, x0_pred = scheduler.sample_prev_timestep(xt, noise_pred, torch.as_tensor(i).to(device))

        # Save x0
        # ims = torch.clamp(xt, -1., 1.).detach().cpu()
        if i == 0:
            # Decode ONLY the final image to save time
            ims = decoder(xt).to(device)
        else:
            ims = xt

        ims = torch.clamp(ims, -1., 1.).detach().cpu()
        ims = (ims + 1) / 2
        grid = make_grid(ims, nrow=train_config['num_grid_rows'])
        img = torchvision.transforms.ToPILImage()(grid)

        if not os.path.exists(os.path.join(train_config['task_name'], 'samples_controlnet')):
            os.mkdir(os.path.join(train_config['task_name'], 'samples_controlnet'))
        img.save(os.path.join(train_config['task_name'], 'samples_controlnet', 'x0_{}.png'.format(i)))
        img.close()


def infer(args):
    # Read the config file #
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    print(config)
    ########################
    model_file = "../model/weight/v1-5-pruned.ckpt"


    models = preload_models_from_standard_weights(model_file, device)
    diffusion_config = config['diffusion_params']
    dataset_config = config['dataset_params']
    diffusion_model_config = config['ldm_params']
    autoencoder_model_config = config['autoencoder_params']
    train_config = config['train_params']

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

    # Create the noise scheduler
    scheduler = LinearScheduler(num_timesteps=diffusion_config['num_timesteps'],
                                beta_start=diffusion_config['beta_start'],
                                beta_end=diffusion_config['beta_end'])

    celebA = CelebDataset(split='train',
                              im_path=dataset_config['im_path'],
                              im_size=dataset_config['im_size'],
                              im_channels=dataset_config['im_channels'],
                              shuffle = train_config['shuffle'],
                              subset = train_config['subset'],
                              use_latents=False,
                              latent_path=None,
                              return_hint=True,
                              condition_config=condition_config)

    # Instantiate the model

    model = ControlNet(im_channels=autoencoder_model_config['z_channels'],
                       model_config=diffusion_model_config,
                       model_locked=True,
                       model_ckpt=model_file,
                       device=device).to(device)
    model.eval()

    assert os.path.exists(os.path.join(train_config['task_name'],
                                       train_config['controlnet_ckpt_name'])), "Train ControlNet first"
    model.load_state_dict(torch.load(os.path.join(train_config['task_name'],
                                                  train_config['controlnet_ckpt_name']),
                                     map_location=device))
    print('Loaded controlnet checkpoint')

    encoder = models['encoder']
    encoder.eval()
    decoder = models['decoder']
    decoder.eval()
    clip = models['clip']
    clip.eval()

    # Load vae if found
    # assert os.path.exists(os.path.join(train_config['task_name'], train_config['vae_autoencoder_ckpt_name'])), \
    #     "VAE checkpoint not present. Train VAE first."
    # vae.load_state_dict(torch.load(os.path.join(train_config['task_name'],
    #                                             train_config['vae_autoencoder_ckpt_name']),
    #                                map_location=device), strict=True)
    print('Loaded vae checkpoint')

    with torch.no_grad():
        sample(model, scheduler, train_config, diffusion_model_config, condition_types, tokenizer, condition_config,
               autoencoder_model_config, diffusion_config, dataset_config, encoder, decoder, clip, celebA)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for ldm controlnet training')
    parser.add_argument('--config', dest='config_path',
                        default='../config/vehicle.yaml', type=str)
    args = parser.parse_args()
    infer(args)