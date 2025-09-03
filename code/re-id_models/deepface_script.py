from deepface import DeepFace
import os
import pandas as pd
from tqdm.auto import tqdm

def test_images(img1_path, img2_path):
  r = DeepFace.verify(
  img1_path = img1_path,
  img2_path = img2_path,
  )

  return str(r['verified']), str(r['distance'])