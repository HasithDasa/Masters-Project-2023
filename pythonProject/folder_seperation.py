import cv2
import os
import shutil


parent_dir = "D:/Academic/MSc/5th Semester/Project/pythonProject"
folder_names = ["final_images_ok", "final_images_not"]

for folder_name in folder_names:
    folder_path = os.path.join(parent_dir, folder_name)
    if os.path.isdir(folder_path):
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            try:
                if os.path.isdir(file_path):
                    # delete subdirectory and its contents recursively
                    os.removedirs(file_path)
                else:
                    # delete file
                    os.remove(file_path)
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")




# Path to the folder containing images
image_folder = "D:/Academic/MSc/5th Semester/Project/pythonProject/final_images"

# Path to the folder where OK images will be moved
ok_folder = "D:/Academic/MSc/5th Semester/Project/pythonProject/final_images_ok"

# Path to the folder where not-OK images will be moved
not_ok_folder = "D:/Academic/MSc/5th Semester/Project/pythonProject/final_images_not"

# Loop through all images in the folder
for filename in os.listdir(image_folder):
    # Load image
    img = cv2.imread(os.path.join(image_folder, filename))

    scale_percent = 50  # percent of original size
    width = int(2048 * scale_percent / 100)
    height = int(1536 * scale_percent / 100)
    dim = (width, height)

    # Resized images for display
    resized_image = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    # Show image
    cv2.imshow(filename, resized_image)

    # Wait for key press
    key = cv2.waitKey(0)

    # Check which key was pressed
    if key == ord('k'):
        # Move image to OK folder
        os.rename(os.path.join(image_folder, filename), os.path.join(ok_folder, filename))
    elif key == ord('n'):
        # Move image to not-OK folder
        os.rename(os.path.join(image_folder, filename), os.path.join(not_ok_folder, filename))
    elif key == ord('e'):
        # Delete image
        os.remove(os.path.join(image_folder, filename))

    # Close image window
    cv2.destroyAllWindows()



# Specify the paths of the source and destination folders
dest_folder = "D:/Academic/MSc/5th Semester/Project/pythonProject/55"

# Use os.makedirs() function to create the destination folder
os.makedirs(dest_folder, exist_ok=True)

# Use the shutil module's copy() function to copy the source folder to the destination folder
shutil.copytree(ok_folder, os.path.join(dest_folder, os.path.basename(ok_folder)))

# Use the shutil module's copy() function to copy the source folder to the destination folder
shutil.copytree(not_ok_folder, os.path.join(dest_folder, os.path.basename(not_ok_folder)))