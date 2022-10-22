import bentoml
from bentoml.io import Image, JSON, Multipart
import torch

from schemas import Txt2ImgInput
from schemas import Img2ImgInput
from stablediffusion import StableDiffusionRunnable


stable_diffusion_runner = bentoml.Runner(StableDiffusionRunnable, name='stable_diffusion_runner', max_batch_size=10)

svc = bentoml.Service("stable_diffusion", runners=[stable_diffusion_runner])


def generate_seed_if_needed(seed):
    if seed is None:
        seed = torch.seed()
    return seed


@svc.api(input=JSON(pydantic_model=Txt2ImgInput), output=Image())
def text2image(data, context):
    data = data.dict()
    data['seed'] = generate_seed_if_needed(data['seed'])
    image = stable_diffusion_runner.txt2img.run(data)
    for i in data:
        context.response.headers.append(i, str(data[i]))
    return image


img2img_input_spec = Multipart(img=Image(), data=JSON(pydantic_model=Img2ImgInput))


@svc.api(input=img2img_input_spec, output=Image())
def image2image(img, data, context):
    data = data.dict()
    data['seed'] = generate_seed_if_needed(data['seed'])
    image = stable_diffusion_runner.img2img.run(img, data)
    for i in data:
        context.response.headers.append(i, str(data[i]))
    return image


inpaint_input_spec = Multipart(img=Image(), mask=Image(), data=JSON(pydantic_model=Img2ImgInput))


@svc.api(input=inpaint_input_spec, output=Image())
def inpainting(img, mask, data, context):
    data = data.dict()
    data['seed'] = generate_seed_if_needed(data['seed'])
    image = stable_diffusion_runner.inpaint.run(img, mask, data)
    for i in data:
        context.response.headers.append(i, str(data[i]))
    return image
