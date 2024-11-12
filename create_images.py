from pathlib import Path
from tqdm import tqdm
import cv2
import os
import random
from PIL import Image
from joblib import Parallel, delayed



def load_images(n_samples: int, video_path: str, resize: tuple[int, int]):
    """Extract and save randomly chosen frames in parallel."""

    def save_frame(idx, frame_id):
        """Extract and save a chosen video frame."""

        # Initialize the video to the frame
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)

        # Load the chosen video frame using opencv
        ret, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # resize and save the image using pillow
        frame = Image.fromarray(frame).resize(resize)
        frame.save(image_path / f"{idx:05d}.jpg")


    # Load the video and initialize the frames
    video = cv2.VideoCapture(video_path)
    total = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = random.sample(range(total), n_samples)

    # Check whether the images are already loaded
    if not os.listdir(image_path):
        enumerator = tqdm(enumerate(indices), total=n_samples)
        Parallel(n_jobs=-1)(delayed(save_frame)(x, y) for x, y in enumerator)


if __name__ == "__main__":
    video_path = 'videos/plastic_bag.mp4'
    storage_path = 'datasets/plastic_bag'

    # Create the storage of the frames and the masks
    storage_path = Path(storage_path)
    image_path = storage_path / "images"
    masks_path = storage_path / "masks"

    image_path.mkdir(parents=True, exist_ok=True)
    masks_path.mkdir(parents=True, exist_ok=True)
    storage_path.mkdir(parents=True, exist_ok=True)


    load_images(5, video_path, (1024, 1024))