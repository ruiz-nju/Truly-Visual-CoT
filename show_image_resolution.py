from PIL import Image
import os

# Specify your folder path here
folder_path = "/mnt/hdd/zhurui/code/Truly-Visual-CoT/data/mathvision/images"

# Supported image file extensions
image_extensions = (".png", ".jpg", ".jpeg", ".bmp", ".gif")

# List to store (filename, width, height)
image_info = []

# Loop through all files in the folder
for filename in os.listdir(folder_path):
    if filename.lower().endswith(image_extensions):
        file_path = os.path.join(folder_path, filename)
        try:
            with Image.open(file_path) as img:
                width, height = img.size
                image_info.append((filename, width, height))
        except Exception as e:
            print(f"Failed to open {filename}: {e}")

# Sort by resolution (total pixel count: width × height)
image_info.sort(key=lambda x: x[1] * x[2])

# Print sorted result
if image_info:
    print("Images sorted by resolution (from smallest to largest):")
    for name, w, h in image_info:
        print(f"{name}: {w}x{h} = {w*h} pixels")
else:
    print("No image files found in the specified folder.")
