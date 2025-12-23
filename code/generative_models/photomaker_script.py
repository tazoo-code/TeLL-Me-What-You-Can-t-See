import torch
import numpy as np
import os
from PIL import Image

from diffusers.utils import load_image
from diffusers import EulerDiscreteScheduler, T2IAdapter
from huggingface_hub import hf_hub_download

from photomaker.pipeline_t2i_adapter import PhotoMakerStableDiffusionXLAdapterPipeline
from photomaker.insightface_package import FaceAnalysis2, analyze_faces


base_model_path = 'SG161222/RealVisXL_V3.0'
photomaker_path = hf_hub_download(repo_id="TencentARC/PhotoMaker-V2", filename="photomaker-v2.bin", repo_type="model")
face_detector = FaceAnalysis2(providers=['CPUExecutionProvider', 'CUDAExecutionProvider'], allowed_modules=['detection', 'recognition'])
face_detector.prepare(ctx_id=0, det_size=(640, 640))

try:
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
except:
    device = "cpu"

if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
    torch_dtype = torch.bfloat16
else:
    torch_dtype = torch.float16
    
save_path = "./outputs"
os.makedirs(save_path, exist_ok=True)

adapter = T2IAdapter.from_pretrained(
    "TencentARC/t2i-adapter-sketch-sdxl-1.0", torch_dtype=torch_dtype, variant="fp16"
).to(device)

# Load base model
pipe = PhotoMakerStableDiffusionXLAdapterPipeline.from_pretrained(
    base_model_path, 
    adapter=adapter, 
    torch_dtype=torch_dtype,
    use_safetensors=True, 
    variant="fp16",
).to(device)

# Load PhotoMaker checkpoint
pipe.load_photomaker_adapter(
    os.path.dirname(photomaker_path),
    subfolder="",
    weight_name=os.path.basename(photomaker_path),
    trigger_word="img",
    pm_version="v2",
)
pipe.id_encoder.to(device)

pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
pipe.fuse_lora()
pipe.to(device)


upload_images = [] 
prompt = ""
negative_prompt = ""
## Parameter setting
guidance_scale = 5
num_steps = 50
style_strength_ratio = 20
start_merge_step = int(float(style_strength_ratio) / 100 * num_steps)
if start_merge_step > 30:
    start_merge_step = 30

image_token_id = pipe.tokenizer.convert_tokens_to_ids(pipe.trigger_word)
input_ids = pipe.tokenizer.encode(prompt)
if image_token_id not in input_ids:
    raise ValueError(f"Cannot find the trigger word '{pipe.trigger_word}' in text prompt!")

if input_ids.count(image_token_id) > 1:
    raise ValueError(f"Cannot use multiple trigger words '{pipe.trigger_word}' in text prompt!")

# determine output dimensions by the aspect ratio
output_w, output_h = 1024, 1024

adapter_conditioning_scale = 0.
adapter_conditioning_factor = 0.
sketch_image = None
    
if upload_images is None:
    raise ValueError(f"Cannot find any input face image!")

input_id_images = []
for img in upload_images:
    input_id_images.append(load_image(img))

id_embed_list = []

for img in input_id_images:
    img = np.array(img)
    img = img[:, :, ::-1]
    faces = analyze_faces(face_detector, img)
    if len(faces) > 0:
        id_embed_list.append(torch.from_numpy((faces[0]['embedding'])))

if len(id_embed_list) == 0:
    raise ValueError(f"No face detected, please update the input face image(s)")

id_embeds = torch.stack(id_embed_list)

generator = torch.Generator(device=device).manual_seed(45)

images = pipe(
    prompt=prompt,
    width=output_w,
    height=output_h,
    input_id_images=input_id_images,
    negative_prompt=negative_prompt,
    num_images_per_prompt=1,
    num_inference_steps=num_steps,
    start_merge_step=start_merge_step,
    generator=generator,
    guidance_scale=guidance_scale,
    id_embeds=id_embeds,
    image=sketch_image,
    adapter_conditioning_scale=adapter_conditioning_scale,
    adapter_conditioning_factor=adapter_conditioning_factor,
).images
