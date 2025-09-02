
from PIL import Image

import collections
import importlib
import io, gdown 
import flax, os
import jax.numpy as jnp
import ml_collections
import numpy as np
import tensorflow as tf


_MODEL_FILENAME = 'maxim'

_MODEL_VARIANT_DICT = {
    'Denoising': 'S-3',
    'Deblurring': 'S-3',
    'Deraining': 'S-2',
    'Dehazing': 'S-2',
    'Enhancement': 'S-2',
}

_MODEL_CONFIGS = {
    'variant': '',
    'dropout_rate': 0.0,
    'num_outputs': 3,
    'use_bias': True,
    'num_supervision_scales': 3,
}

class DummyFlags():
  def __init__(self, ckpt_path:str, task:str, input_dir: str = "./maxim/images/Enhancement", output_dir:str = "./maxim/images/Results", has_target:bool = False, save_images:bool = True, geometric_ensemble:bool = False):
    '''
    Builds the dummy flags which replicates the behaviour of Terminal CLI execution (same as ArgParse)
    args:
      ckpt_path: Saved Model CheckPoint: Find all the checkpoints for pre trained models at https://console.cloud.google.com/storage/browser/gresearch/maxim/ckpt/
      task: Task for which the model waas trained. Each task uses different Data and Checkpoints. Find the details of tasks and respective checkpoints details at: https://github.com/google-research/maxim#results-and-pre-trained-models
      input_dir: Input Directory. We do not need it here as we are directly passing one image at a time
      output_dir: Also not needed in out code
      has_target: Used to calculate PSNR and SSIM calculation. Not needed in our case
      save_images: Used in CLI command where images were saved in loop. Not needed in our case
      geometric_ensemble: Was used in training part and as it is just an Inference part, it is not needed

    '''
    self.ckpt_path = ckpt_path
    self.task = task
    self.input_dir = input_dir
    self.output_dir = output_dir
    self.has_target = has_target
    self.save_images = save_images
    self.geometric_ensemble = geometric_ensemble

# Copyright 2022 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


def recover_tree(keys, values):
  """Recovers a tree as a nested dict from flat names and values.

  This function is useful to analyze checkpoints that are saved by our programs
  without need to access the exact source code of the experiment. In particular,
  it can be used to extract an reuse various subtrees of the scheckpoint, e.g.
  subtree of parameters.
  Args:
    keys: a list of keys, where '/' is used as separator between nodes.
    values: a list of leaf values.
  Returns:
    A nested tree-like dict.
  """
  tree = {}
  sub_trees = collections.defaultdict(list)
  for k, v in zip(keys, values):
    if '/' not in k:
      tree[k] = v
    else:
      k_left, k_right = k.split('/', 1)
      sub_trees[k_left].append((k_right, v))
  for k, kv_pairs in sub_trees.items():
    k_subtree, v_subtree = zip(*kv_pairs)
    tree[k] = recover_tree(k_subtree, v_subtree)
  return tree


def mod_padding_symmetric(image, factor=64):
  """Padding the image to be divided by factor."""
  height, width = image.shape[0], image.shape[1]
  height_pad, width_pad = ((height + factor) // factor) * factor, (
      (width + factor) // factor) * factor
  padh = height_pad - height if height % factor != 0 else 0
  padw = width_pad - width if width % factor != 0 else 0
  image = jnp.pad(
      image, [(padh // 2, padh // 2), (padw // 2, padw // 2), (0, 0)],
      mode='reflect')
  return image


def get_params(ckpt_path):
  """Get params checkpoint."""

  with tf.io.gfile.GFile(ckpt_path, 'rb') as f:
    data = f.read()
  values = np.load(io.BytesIO(data))
  params = recover_tree(*zip(*values.items()))
  params = params['opt']['target']

  return params

def make_shape_even(image):
  """Pad the image to have even shapes."""
  height, width = image.shape[0], image.shape[1]
  padh = 1 if height % 2 != 0 else 0
  padw = 1 if width % 2 != 0 else 0
  image = jnp.pad(image, [(0, padh), (0, padw), (0, 0)], mode='reflect')
  return image


# Refactored code --------------------------------------------------------------------------------------------------------------------

def build_model(task = "Enhancement"):
  model_mod = importlib.import_module(f'maxim.models.{_MODEL_FILENAME}')
  model_configs = ml_collections.ConfigDict(_MODEL_CONFIGS)

  model_configs.variant = _MODEL_VARIANT_DICT[task]

  model = model_mod.Model(**model_configs)
  return model


def pre_process(input_file):
  '''
  Pre-process the image before sending to the model
  '''
  input_img = np.asarray(Image.open(input_file).convert('RGB'),np.float32) / 255.
  # Padding images to have even shapes
  height, width = input_img.shape[0], input_img.shape[1]
  input_img = make_shape_even(input_img)
  height_even, width_even = input_img.shape[0], input_img.shape[1]

  # padding images to be multiplies of 64
  input_img = mod_padding_symmetric(input_img, factor=64)
  input_img = np.expand_dims(input_img, axis=0)

  return input_img, height, width, height_even, width_even


def predict(input_img):
  # handle multi-stage outputs, obtain the last scale output of last stage
  return model.apply({'params': flax.core.freeze(params)}, input_img)


def post_process(preds, height, width, height_even, width_even):
  '''
  Post process the image coming out from prediction
  '''
  if isinstance(preds, list):
    preds = preds[-1]
    if isinstance(preds, list):
      preds = preds[-1]

  # De-ensemble by averaging inferenced results.
  preds = np.array(preds[0], np.float32)

  # unpad images to get the original resolution
  new_height, new_width = preds.shape[0], preds.shape[1]
  h_start = new_height // 2 - height_even // 2
  h_end = h_start + height
  w_start = new_width // 2 - width_even // 2
  w_end = w_start + width
  preds = preds[h_start:h_end, w_start:w_end, :]
  return np.array((np.clip(preds, 0., 1.) * 255.).astype(jnp.uint8))


def apply_maxim(path, output_path):
  img,a,b,c,d = pre_process(path)
  enhanced_image_array = post_process(predict(img),a,b,c,d) # Get predictions
  enhanced_pil_image = Image.fromarray(enhanced_image_array) # get PIL image from array
  enhanced_pil_image.save(output_path) # Save the image
  
weight_drive_path = 'https://storage.googleapis.com/gresearch/maxim/ckpt/Enhancement/LOL/checkpoint.npz' 
MODEL_PATH = './maxim.npz' 
if not os.path.exists(MODEL_PATH):
  gdown.download(weight_drive_path, MODEL_PATH, quiet=False) # Download Model weights to your current instance
FLAGS = DummyFlags(ckpt_path = MODEL_PATH, task = "Enhancement") # Path to your checkpoint and task name
params = get_params(FLAGS.ckpt_path) # Parse the config
model = build_model() # Build Model

