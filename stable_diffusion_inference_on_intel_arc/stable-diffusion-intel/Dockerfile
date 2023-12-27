from intel-extension-for-pytorch as baseline

RUN pip install accelerate diffusers transformers

RUN python -c "import torch; from diffusers import StableDiffusionPipeline; StableDiffusionPipeline.from_pretrained('runwayml/stable-diffusion-v1-5');"

