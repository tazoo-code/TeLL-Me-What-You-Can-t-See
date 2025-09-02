from tinyllava.model.builder import load_pretrained_model
from tinyllava.mm_utils import get_model_name_from_path

import argparse
import torch, os
from tqdm.auto import tqdm

from tinyllava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from tinyllava.conversation import conv_templates, SeparatorStyle
from tinyllava.model.builder import load_pretrained_model
from tinyllava.utils import disable_torch_init
from tinyllava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
    KeywordsStoppingCriteria,
)

from PIL import Image

import requests
from PIL import Image
from io import BytesIO
import re

def image_parser(args):
    out = args.image_file.split(args.sep)
    return out


def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def load_images(image_files):
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        out.append(image)
    return out

def eval_model(args, tokenizer, model, image_processor, context_len):
    # Model
    disable_torch_init()

    qs = args.query
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    if IMAGE_PLACEHOLDER in qs:
        if model.config.mm_use_im_start_end:
            qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
        else:
            qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
    else:
        if model.config.mm_use_im_start_end:
            qs = image_token_se + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    image_files = image_parser(args)
    images = load_images(image_files)
    images_tensor = process_images(
        images,
        image_processor,
        model.config
    ).to(model.device, dtype=torch.float16)

    input_ids = (
        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .cuda()
    )

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=images_tensor,
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            pad_token_id=tokenizer.pad_token_id,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
            stopping_criteria=[stopping_criteria],
        )

    outputs = tokenizer.batch_decode(
        output_ids, skip_special_tokens=True)[0]

    outputs = outputs.strip()
    if outputs.endswith(stop_str):
        outputs = outputs[: -len(stop_str)]
    outputs = outputs.strip()
    return outputs


# helper function to build the necessary arguments
def make_args(prompt, image):
  args = type('Args', (), {
    "model_path": model_path,
    "model_base": None,
    "model_name": get_model_name_from_path(model_path),
    "query": prompt,
    "conv_mode": "phi",
    "image_file": image,
    "sep": ",",
    "temperature": 0,
    "top_p": None,
    "num_beams": 1,
    "max_new_tokens": 512
  })()

  return args

def infer_tinyllava(prompts, path):
  output = []
  for prompt in prompts:
    args = make_args(prompt, path)
    output += [eval_model(args, tokenizer, model, image_processor, context_len)]
  return output

model_path = "bczhou/TinyLLaVA-3.1B"

model_name = get_model_name_from_path(model_path)
tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, None, model_name)

prompts = [
  'What is the gender of the subject? Answer with only one word.',
  'What is the age of the subject? Answer with only a number.',
  'Analyzing the physical appearance of the subject, which ethnical group may he/she belong to? Answer with only one word.',
  'What is the hair color of the subject? Answer with only one word.',
  'Are the hair long or short? Answer with only one word.',
  'Are they straight, wavy, curly or kinky? Answer with only one word.',
  'Does the spacing between the eyes appear to be wide, medium, or narrow? Answer with only one word.',
  'Do the eyebrows rest high above the eyes, down low over the eyes, or in between? Choose only one option. Answer with only one word.',
  'Are the eyebrows rather straight across or are they noticeably arched? Answer with only one word.',
  'Is the upper eyelid is highly, mediumly, lowly or not visible? Answer with only one word.',
  'Is there an epicanthic fold on the eyes? Answer with only one word.',
  'Is the "look" of the eyelids alert and wide open or sleepy and semi-closed? Choose only one option. ',
  'Is the sclera very white, or is it yellow or reddish?  Answer with only one word.',
  'What is the color of the iris? Use at most two words.',
  'Is the area around the eyes puffy, dark or wrinkled? Answer with only one word.',
  'Are the lashes very visible? Answer with only one word.',
  'What is the expression in the eyes? Answer with only one word.',
  'Does the nasal bridge seem wide, medium, or narrow? Answer with only one word.',
  'Is the nose upturned, straight out, or downturned? Answer with only one word.',
  'What is the shape of the tip of the nose? Answer with only one word.',
  'Does the mouth seem wide, medium, or narrow? Answer with only one word.',
  'Does the mouth vertical thickness seem wide, medium, or narrow? Answer with only one word.',
  'Are the lips equally thick, thicker on top, or thicker on bottom? Choose only one option. ',
  'Are the teeth visible? Answer with only one word.',
  'Are the lips color dark or pale? Answer with only one word.',
  'Are there facial hair? Which is their color? How long are they? Give a short answer.',
  'Are the ears large, medium or small? Answer with only one word.',
  'Are the ears protruding? Answer with only one word.',
  'Can you see inside the ears a bit? Answer with only one word.',
  'Do the ears stick out more at the top or at the bottom? Choose only one option.',
  'How much tall is the person in the image? Answer, even if you are not able nor accurate, with a just reasonable estimate of the height expressed in foot.',
  'What is the weight of the person in the image in pounds? Answer with just an estimate of the weight expressed in pounds.',
  'Does the person in the image have tattoos? If so where are they exactly located? Give a short answer.',
  'Does the person in the image have evident moles? If so where are they exactly located? Give a short answer.',
  'Does the person in the image have scars? If so where are they exactly located? Give a short answer.',
  'Are the subject wearing makeup? Which type? Give a short answer.',
  'Please describe what is the person in the image wearing. Give a short answer.']





