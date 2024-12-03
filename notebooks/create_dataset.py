import cv2
import random

from pathlib import Path
from tqdm import tqdm
from PIL import Image
import pandas as pd


def create_dataset(metadata: pd.DataFrame, videos_path: Path, image_path: Path, n_samples: int) -> None:
    """Extract and save randomly chosen frames from multiple videos.

    :param metadata: DataFrame containing metadata about each video.
    :param videos_path: Directory containing the videos.
    :param image_path: Directory where the images will be saved.
    :param n_samples: Number of randomly chosen frames per video"""

    start_idx = 0
    resize = (1024, 1024)
    image_metadata = []

    for _, row in tqdm(metadata.iterrows(), desc='processing videos'):
        # Load the video and chose random frames
        video_path = videos_path / row['video_names']
        video = cv2.VideoCapture(video_path)

        total = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        indices = random.sample(range(total), n_samples)

        # Check whether we have to rotate the frames
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        rotate = (width > height)

        for idx, frame_id in enumerate(indices):
            # Load the chosen frame using opencv
            video.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            ret, frame = video.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Resize and rotate the frame if necessary
            frame = Image.fromarray(frame).resize(resize)
            frame.rotate(-90, expand=True) if rotate else frame

            # Save the frame using pillow
            image_name = f"{(start_idx + idx):05d}.jpg"
            frame.save(image_path / 'images' / image_name)

            # Include the image name and its class
            image_row = {'image_name': image_name, 'class': row['class']}
            image_metadata.append(image_row)
        start_idx += n_samples

    # Save to CSV or explore the DataFrame
    image_metadata = pd.DataFrame(image_metadata)
    image_metadata.to_csv(image_path / "metadata.csv", index=False)



if __name__ == "__main__":
    videos_path = Path('../videos')
    storage_path = Path('../datasets/training')
    metadata = pd.read_csv('../datasets/training.csv')

    storage_path.mkdir(parents=True, exist_ok=True)
    (storage_path / 'images').mkdir(parents=True, exist_ok=True)
    (storage_path / 'masks').mkdir(parents=True, exist_ok=True)

    create_dataset(metadata, videos_path, storage_path, 10)

