import os
import glob
from PIL import Image
import math
import sys

# --- Configuration ---
# NOTE ON COMPRESSION: To achieve a noticeable file size reduction (like 50%),
# we MUST switch to a lossy format like JPEG (.jpg) and use the 'quality' parameter.
# PNG is lossless, yielding minimal file size changes on already-optimized images.
# Allow passing the target directory as the first CLI argument:
# Usage: python merge-png.py /path/to/dir
if len(sys.argv) > 1:
    DIRECTORY = sys.argv[1]
else:
    exit()
OUTPUT_FILENAME = f"{DIRECTORY}/collage_output.jpg"  # Changed to JPEG

JPEG_QUALITY = 50 # Set the compression quality (0=max compression/min quality, 95=min compression/max quality)
PADDING = 1    # User-requested spacing

def create_collage(directory=f"{DIRECTORY}"):
    """
    Finds all PNG files in the specified directory, determines the largest
    dimensions to preserve resolution, resizes them, arranges them into
    a single tiled collage image with minimal padding, and saves the final
    image as a compressed JPEG.
    """
    # Ensure the output directory exists before proceeding
    output_dir = os.path.dirname(OUTPUT_FILENAME)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    print(f"Starting collage creation in directory: {os.path.abspath(directory)}")
    
    # 1. Find all PNG files and determine required tile size
    # Note: We must look for PNG files in the input directory, regardless of the output filename's extension
    png_files = [f for f in glob.glob(os.path.join(directory, "*.png"))]

    if not png_files:
        print("No .png files found to create a collage. Exiting.")
        return

    print(f"Found {len(png_files)} image(s): {', '.join(png_files)}")

    # Temporary list to hold loaded images and determine max dimensions
    loaded_images = []
    max_width = 0
    max_height = 0

    for file in png_files:
        try:
            # Load as RGB. While the source is PNG (RGBA), JPEG doesn't support transparency,
            # so converting to RGB here avoids warnings and prepares for JPEG saving.
            img = Image.open(file).convert("RGB") 
            loaded_images.append(img)
            # Determine the maximum required tile dimensions
            max_width = max(max_width, img.width)
            max_height = max(max_height, img.height)
        except Exception as e:
            print(f"Warning: Could not load or process {file}. Skipping. Error: {e}")

    if not loaded_images:
        print("No usable images were loaded. Exiting.")
        return

    # Define the final tile dimensions based on the largest image found
    TILE_WIDTH = max_width
    TILE_HEIGHT = max_height
    images = []

    # 2. Resize all images to the determined max dimensions (ensures a uniform grid)
    print(f"Using a uniform tile size of {TILE_WIDTH}x{TILE_HEIGHT} (based on largest image found).")
    for img in loaded_images:
        img = img.resize((TILE_WIDTH, TILE_HEIGHT))
        images.append(img)

    # 3. Determine grid dimensions
    num_images = len(images)
    cols = math.ceil(math.sqrt(num_images))
    rows = math.ceil(num_images / cols)

    print(f"Arranging images in a {rows}x{cols} grid.")

    # 4. Calculate final canvas size
    collage_width = (cols * TILE_WIDTH) + ((cols + 1) * PADDING)
    collage_height = (rows * TILE_HEIGHT) + ((rows + 1) * PADDING)

    # 5. Create the new canvas (using RGB, as we are saving to JPEG)
    collage = Image.new('RGB', (collage_width, collage_height), (255, 255, 255)) # White background

    # 6. Paste images onto the canvas
    for index, img in enumerate(images):
        # Calculate current position in the grid
        r = index // cols
        c = index % cols

        # Calculate coordinates for pasting
        x_offset = PADDING + c * (TILE_WIDTH + PADDING)
        y_offset = PADDING + r * (TILE_HEIGHT + PADDING)

        # Paste the resized image
        collage.paste(img, (x_offset, y_offset))

    # 7. Save the final image with lossy JPEG compression
    collage.save(OUTPUT_FILENAME, quality=JPEG_QUALITY)

    print(f"\nSuccessfully created collage: {OUTPUT_FILENAME}")
    print(f"Final Collage size: {collage_width}x{collage_height} pixels. Saved as JPEG with quality={JPEG_QUALITY}.")


if __name__ == "__main__":
    create_collage()
