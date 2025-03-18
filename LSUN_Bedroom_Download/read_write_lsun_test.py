import os
from PIL import Image, UnidentifiedImageError
from concurrent.futures import ThreadPoolExecutor, as_completed

# Define the directory containing the image files
image_directory = '/dataset/LSUN_Bedroom/train_orig_imgs'

# Define the file to write the problematic files
error_log_file = 'error_files.txt'

# Function to check if an image can be opened
def check_image(file_path):
    try:
        with Image.open(file_path) as img:
            img.verify()  # Verify that it is, in fact, an image
        return None
    except (IOError, UnidentifiedImageError):
        return file_path

# Function to write errors to a file
def log_error(file_path):
    with open(error_log_file, 'a') as file:
        file.write(f"{file_path}\n")

# Get a list of image files
image_files = [os.path.join(image_directory, f) for f in os.listdir(image_directory)
               if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]

# Use ThreadPoolExecutor to check images in parallel
with ThreadPoolExecutor() as executor:
    # Map the check_image function to the image files
    future_to_image = {executor.submit(check_image, file_path): file_path for file_path in image_files}
    for future in as_completed(future_to_image):
        file_path = future_to_image[future]
        try:
            result = future.result()
            if result is not None:
                # If result is not None, it means there was an error with the image
                log_error(result)
        except Exception as exc:
            print(f"{file_path} generated an exception: {exc}")

print("Image verification completed.")
