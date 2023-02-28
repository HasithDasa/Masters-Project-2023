import os
import openpyxl

# Create a new workbook
workbook = openpyxl.Workbook()
sheet = workbook.active

# Define the path to the folder containing the subfolders
path = "D:/Academic/MSc/5th Semester/Project/pythonProject/Results"

# Get the list of folders in the path directory
folders = os.listdir(path)

# Iterate through each folder
for folder_name in folders:
    # Check if folder_name is a folder and not a file
    if os.path.isdir(os.path.join(path, folder_name)):
        # Get the path to the folder
        folder_path = os.path.join(path, folder_name)

        # Get the number of files in final_images_ok folder
        ok_files = os.listdir(os.path.join(folder_path, "final_images_ok"))
        num_ok_files = len([file for file in ok_files if file.endswith(".jpg")])

        # Get the number of files in final_images_not folder
        not_files = os.listdir(os.path.join(folder_path, "final_images_not"))
        num_not_files = len([file for file in not_files if file.endswith(".jpg")])

        # Add the folder name and file count to the sheet
        sheet.append([folder_name, num_ok_files, num_not_files])

# Save the workbook
workbook.save("folder_file_counts.xlsx")
