import cv2
import os
from PIL import Image
from pathlib import Path
from tqdm import tqdm

directory = Path('datasets/C1H0_CAM01')
new_images = directory / 'new_images'
new_images.mkdir(parents=True, exist_ok=True)
old_images = directory / 'images'
image_names = os.listdir(old_images)


for image_name in tqdm(image_names, total=len(image_names)):
    image_path = old_images / image_name
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    image = image.rotate(-90, expand=True)
    image.save(new_images / image_name)


