import glob
import os
from PIL import Image

import h3

def generate_gif(directory):
    """Generate gif based on a list of numbered png"""
    if not directory.endswith('/'): directory += '/'
    png_files = sorted(glob.glob(f"{directory}*.png"))
    sorted_png_files = sorted(png_files, key=lambda x: int(x.split('/')[2].split('.')[0]))

    images = []
    for filename in sorted_png_files:
        img = Image.open(filename).convert('RGBA')
        images.append(img)
    output_gif_path = f"{directory}Spiral_viz.gif"
    images[0].save(output_gif_path, save_all=True, append_images=images[1:], optimize=False, duration=200, loop=0)

    print(f"Generated GIF saved at {output_gif_path}")

if __name__ == "__main__":
    current_working_directory = os.getcwd()

    print(current_working_directory)
    generate_gif("./gif/")