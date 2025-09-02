from huggingface_hub import snapshot_download
import torch
import subprocess
from PIL import Image
import sys


repo_path = snapshot_download(
    repo_id="thiemcun203/super-resolution", 
    repo_type="space"
)

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-r", str(f'{repo_path}/requirements.txt')])
sys.path.append(f'{repo_path}/models/SRGAN')
from srgan import GeneratorResnet, ResidualBlock


def apply_srgan(path, output_path):
    image = Image.open(path).convert("RGB")
    srgan_model = GeneratorResnet()
    device = torch.device('cpu') if not torch.cuda.is_available() else torch.device('cuda')
    srgan_model = torch.load(f'{repo_path}/models/SRGAN/srgan_checkpoint.pth', map_location=device, weights_only=False)

    enhanced_image = srgan_model.inference(image)
    enhanced_image.save(output_path)

