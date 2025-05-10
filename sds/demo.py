import model_loader
import pipeline
from PIL import Image
from pathlib import Path
from transformers import CLIPTokenizer
import torch
import matplotlib.pyplot as plt
DEVICE = "cpu"

ALLOW_CUDA = True
ALLOW_MPS = True

if torch.cuda.is_available() and ALLOW_CUDA:
    DEVICE = "cuda"
elif (torch.has_mps or torch.backends.mps.is_available()) and ALLOW_MPS:
    DEVICE = "mps"
print(f"Using device: {DEVICE}")

tokenizer = CLIPTokenizer("../data/tokenizer_vocab.json", merges_file="../data/tokenizer_merges.txt")
model_file = "../model/weight/v1-5-pruned.ckpt"
models = model_loader.preload_models_from_standard_weights(model_file, DEVICE)

## TEXT TO IMAGE

# prompt = "A dog with sunglasses, wearing comfy hat, looking at camera, highly detailed, ultra sharp, cinematic, 100mm lens, 8k resolution."
# prompt = "A majestic female a dog character with a long a long dark hair adorned with golden ornaments wearing \
#  a flowing white dress with gold accents and large white feathered wings, gracefully soaring through a stormy sky\
#   , illuminated by a brilliant flash of lightning,  camera view from below"

# prompt = "remove the lion highlighted in blue or remove the masked object, or delete the object highlighted in blue or the masked objet"

for i in range(25):
    uncond_prompt = ""  # Also known as negative prompt
    prompt = "8k resolution, highly detailed, ultra sharp, cinematic, 100mm lens."

    # do_cfg = True
    do_cfg = True
    cfg_scale = 0  # min: 1, max: 14

    ## IMAGE TO IMAGE

    # input_image = None
    # Comment to disable image to image
    image_path = "editable/Picture" + str(i+1)+".png"
    input_image = Image.open(image_path)
    # Higher values means more noise will be added to the input image, so the result will further from the input image.
    # Lower values means less noise is added to the input image, so output will be closer to the input image.
    strength = 0.1

    ## SAMPLER

    sampler = "ddpm"
    num_inference_steps = 50
    seed = 42

    output_image = pipeline.generate(
        prompt=prompt,
        uncond_prompt=uncond_prompt,
        input_image=input_image,
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
    img.save("../sds/editable/recon"+str(i+1)+".png")
    img.show()