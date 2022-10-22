from contextlib import ExitStack

import bentoml
from diffusers import StableDiffusionImg2ImgPipeline
from diffusers import StableDiffusionInpaintPipeline
from diffusers import StableDiffusionPipeline
import torch
from torch import autocast


model_id = 'CompVis/stable-diffusion-v1-4'
hf_auth_token = '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'


class StableDiffusionRunnable(bentoml.Runnable):
    SUPPORTED_RESOURCES = ("nvidia.com/gpu", "cpu")
    SUPPORTS_CPU_MULTI_THREADING = True

    def __init__(self):

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        txt2img_pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            use_auth_token=hf_auth_token,
            revision="fp16",
            torch_dtype=torch.float16,
        )

        self.txt2img_pipe = txt2img_pipe.to(self.device)

        self.img2img_pipe = StableDiffusionImg2ImgPipeline(
            vae=self.txt2img_pipe.vae,
            text_encoder=self.txt2img_pipe.text_encoder,
            tokenizer=self.txt2img_pipe.tokenizer,
            unet=self.txt2img_pipe.unet,
            scheduler=self.txt2img_pipe.scheduler,
            safety_checker=self.txt2img_pipe.safety_checker,
            feature_extractor=txt2img_pipe.feature_extractor,
        ).to(self.device)

        self.inpaint_pipe = StableDiffusionInpaintPipeline(
            vae=self.txt2img_pipe.vae,
            text_encoder=self.txt2img_pipe.text_encoder,
            tokenizer=self.txt2img_pipe.tokenizer,
            unet=self.txt2img_pipe.unet,
            scheduler=self.txt2img_pipe.scheduler,
            safety_checker=self.txt2img_pipe.safety_checker,
            feature_extractor=self.txt2img_pipe.feature_extractor,
        ).to(self.device)

    @bentoml.Runnable.method(batchable=False, batch_dim=0)
    def txt2img(self, data):
        prompt = data["prompt"]
        guidance_scale = data.get('guidance', 7.5)
        height = data.get('height', 512)
        width = data.get('width', 512)
        num_inference_steps = data.get('steps', 50)
        generator = torch.Generator(self.device)
        generator.manual_seed(data.get('seed'))

        with ExitStack() as stack:
            if self.device != "cpu":
                _ = stack.enter_context(autocast(self.device))

            images = self.txt2img_pipe(
                prompt=prompt,
                guidance_scale=guidance_scale,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                generator=generator
            ).images
            return images[0]

    @bentoml.Runnable.method(batchable=False, batch_dim=0)
    def img2img(self, init_image, data):
        new_size = None
        longer_side = max(*init_image.size)
        if longer_side > 512:
            new_size = (512, 512)
        elif init_image.width != init_image.height:
            new_size = (longer_side, longer_side)

        if new_size:
            init_image = init_image.resize(new_size)

        prompt = data["prompt"]
        strength = data.get('strength', 0.8)
        guidance_scale = data.get('guidance', 7.5)
        num_inference_steps = data.get('steps', 50)
        generator = torch.Generator(self.device)
        generator.manual_seed(data.get('seed'))

        with ExitStack() as stack:
            if self.device != "cpu":
                _ = stack.enter_context(autocast(self.device))

            images = self.img2img_pipe(
                prompt=prompt,
                init_image=init_image,
                strength=strength,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                generator=generator,
            ).images
            return images[0]

    @bentoml.Runnable.method(batchable=False, batch_dim=0)
    def inpaint(self, image, mask, data):
        prompt = data["prompt"]
        strength = data.get('strength', 0.8)
        guidance_scale = data.get('guidance', 7.5)
        num_inference_steps = data.get('steps', 50)
        generator = torch.Generator(self.device)
        generator.manual_seed(data.get('seed'))

        with ExitStack() as stack:
            if self.device != "cpu":
                _ = stack.enter_context(autocast(self.device))

            images = self.inpaint_pipe(
                prompt=prompt,
                init_image=image,
                mask_image=mask,
                strength=strength,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
            ).images
            image = images[0]
            return image
