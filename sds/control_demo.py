import model_loader
import control_pipline
from PIL import Image
from torchvision.utils import make_grid
import torchvision
from transformers import CLIPTokenizer
import control_model_loader
from utils.config_utils import *
from datasets.celeb_dataset import CelebDataset
import torch
from datasets.vehicle_dataset import VehicleDataset
import random
from controlnet import ControlNet
import argparse
import yaml
import os
def infer(args):
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    print(config)

    ALLOW_CUDA = True
    ALLOW_MPS = False

    if torch.cuda.is_available() and ALLOW_CUDA:
        DEVICE = "cuda"
    elif (torch.has_mps or torch.backends.mps.is_available()) and ALLOW_MPS:
        DEVICE = "mps"
    print(f"Using device: {DEVICE}")

    model_file = "../model/weight/v1-5-pruned.ckpt"
    # model_file = "D:/new_folder/JOSEPH_FILe/python/AdvancedAI/AI/Stable_Diffusion/models/weights/sd-v1-4.ckpt"
    models = model_loader.preload_models_from_standard_weights(model_file, DEVICE)

    # Read the config file #
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
    im_dataset_cls = {
        # 'mnist': MnistDataset,
        'celebhq': CelebDataset,
        'vehicle': VehicleDataset
    }.get(dataset_config['name'])
    dataset = im_dataset_cls(split='train',
                          im_path=dataset_config['im_path'],
                          im_size=dataset_config['im_size'],
                          im_channels=dataset_config['im_channels'],
                          shuffle=train_config['shuffle'],
                          subset=10,
                          use_latents=False,
                          latent_path=None,
                          return_hint=True,
                          condition_config=condition_config)

    # Instantiate the model

    controlnet = ControlNet(im_channels=autoencoder_model_config['z_channels'],
                       model_config=diffusion_model_config,
                       model_locked=True,
                       model_ckpt=model_file,
                       device=DEVICE).to(DEVICE)
    controlnet.eval()

    assert os.path.exists(os.path.join(train_config['task_name'],
                                       train_config['controlnet_ckpt_name'])), "Train ControlNet first"
    controlnet.load_state_dict(torch.load(os.path.join(train_config['task_name'],
                                                  train_config['controlnet_ckpt_name']),
                                     map_location=DEVICE))
    print('Loaded controlnet checkpoint')

    hints = []
    for idx in range(train_config['num_samples']):
        hint_idx = random.randint(0, len(dataset) - 1)  # Fix: Subtract 1 to avoid out-of-bounds
        _, hint, cond_input = dataset[hint_idx]  # Extract only the hint (ignore image and text)
        hints.append(hint.unsqueeze(0).to(DEVICE))  # Add batch dim and move to device
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
    # Concatenate all hints into a single tensor
    hints = torch.cat(hints, dim=0).to(DEVICE)
    print(prompt)
    # Output shape: [num_samples, C, H, W]
    print(hints.shape)
    # Save the hints
    hints_grid = make_grid(hints, nrow=train_config['num_grid_rows'])
    hints_img = torchvision.transforms.ToPILImage()(hints_grid)
    hints_img.save(os.path.join(train_config['task_name'], 'hint.png'))

    uncond_prompt = ""  # Also known as negative prompt
    # do_cfg = True
    do_cfg = False
    cfg_scale = 6  # min: 1, max: 14

    strength = 0.6

    ## SAMPLER

    sampler = "ddpm"
    num_inference_steps = 500
    seed = 42

    output_image = control_pipline.generate(
        prompt=prompt,
        uncond_prompt=uncond_prompt,
        controlnet=controlnet,
        hint= hints,
        input_image=None,
        strength=strength,
        do_cfg=do_cfg,
        sampler_name=sampler,
        n_inference_steps=num_inference_steps,
        seed=seed,
        models=models,
        device=DEVICE,
        idle_device="cpu",
        tokenizer=tokenizer,
    )

    # Combine the input image and the output image into a single image.
    Image.fromarray(output_image)
    img = Image.fromarray(output_image)

    # Save the image (optional)
    # img.save("../image/output_joseph.jpg")
    img.save(os.path.join(train_config['task_name'], 'image_recon.png'))
    print('done')
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for ldm controlnet training')
    parser.add_argument('--config', dest='config_path',
                        default='../config/vehicle.yaml', type=str)
    args = parser.parse_args()
    infer(args)
