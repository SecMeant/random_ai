#!/bin/python
import intel_extension_for_pytorch as ipex
import torch
from diffusers import StableDiffusionPipeline

from sys import argv

model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("xpu")

def gen_image(pipeline, prompt):
    image = pipeline(prompt).images[0]
    image.save("image.png")

gen_image(pipe, argv[1])

