import fire
from pathlib import Path
import tqdm
from PIL import Image, UnidentifiedImageError
from torchvision.transforms.functional import resize

import torch
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode


def get_image_file_names(path: Path):
    image_files = [
        *path.glob("**/*.png"),
        *path.glob("**/*.jpg"),
        *path.glob("**/*.jpeg"),
        *path.glob("**/*.bmp"),
    ]
    return image_files


def main(src: str, dst: str, image_size: int=384, output_extension='.png'):
    
    source_path = Path(src)
    destination_path = Path(dst)

    if not destination_path.exists():
        destination_path.mkdir(parents=True)

    image_paths = get_image_file_names(source_path)
    i = 0
    for source_filename in tqdm.tqdm(image_paths, desc='transforming'):
        # load image
        try:
            img = Image.open(source_filename).convert('RGB')
        except (UnidentifiedImageError, OSError):
            print(f"Failed to load image '{image_path}'. Skipping.")
            continue

        # transform image
        img = resize(img, (image_size,image_size), interpolation=InterpolationMode.BICUBIC)

        # save as png
        destination_filename = source_filename.stem + output_extension
        destination_filename = destination_path / destination_filename

        # img = img.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
        # im = Image.fromarray(img)
        img.save(destination_filename, format=None)
        i += 1

    print(f'{i} image files pre-processed.')

if __name__ == '__main__':
    fire.Fire(main)