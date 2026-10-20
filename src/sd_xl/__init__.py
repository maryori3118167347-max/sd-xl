import torch
from diffusers import AutoPipelineForText2Image

print("Cargando modelo...")

modelo = AutoPipelineForText2Image.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0",torch_dtype=torch.float16,low_cpu_mem_usage=True )
modelo = modelo.to("cpu")
# cuda - nvidia
# mps - mac com procesadores mx
# xpu - grafica intal iris

prompt = input ("Escribe el prompt de la imagen ... ")

negative_prompt = "blurry, low quality, worst quality, distorted face, bad anatomy, extra limbs, extra fingers, poorly drawn hands, watermark, signature, text, error"
print("Generando la imagen")
imagen = modelo (
    prompt = prompt,
    negative_prompt=negative_prompt,
    num_inference_steps=10,# Numero de veces que creara y
    guidance_scale=2.0,# que tan fiel sera al prompt
    height=1024,
    width=1024,


    ).images[0]

imagen.save("imagen.png ")
print("Imagen guardada ! ")
                   
            
