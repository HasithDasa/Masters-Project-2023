import os

parent_dir = "D:/Academic/MSc/5th Semester/Project/pythonProject"
folder_names = ["final_images", "final_images_lines", "final_images_3", "final_images_4", "final_images_3", "final_images_ok", "final_images_not"]

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
