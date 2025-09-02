
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch, os
import pandas as pd
from tqdm.auto import tqdm

torch.manual_seed(1234)

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-VL-Chat-Int4", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat-Int4", device_map="cuda", trust_remote_code=True).eval()

print('here')

def infer_qwen(prompts, path):
  output = []
  for prompt in prompts:
    query = tokenizer.from_list_format([
          {'image': path},
          {'text': prompt},
      ])
    response, history = model.chat(tokenizer, query=query, history=None)
    output += [response]
  return output

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

