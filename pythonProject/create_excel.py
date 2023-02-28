import os
import pandas as pd

path = "D:/Academic/MSc/5th Semester/Project/pythonProject/Results/"
excel_path = "Accuracy_calculation.xlsx"

# Load the Excel file into a pandas DataFrame
df = pd.read_excel(excel_path)

for folder_name in os.listdir(path):
    if os.path.isdir(os.path.join(path, folder_name)):
        name = folder_name
        folder_path = os.path.join(path, name)
        ok_files = os.listdir(os.path.join(folder_path, "final_images_ok"))
        not_ok_files = os.listdir(os.path.join(folder_path, "final_images_not"))
        ok_count = len([f for f in ok_files if f.endswith(".jpg")])
        not_count = len([f for f in not_ok_files if f.endswith(".jpg")])
        print("OK files count:", ok_count)
        print("Not files count:", not_count)

        # Update the DataFrame with the file counts
        df.loc[df["Name"] == name, "final_images_ok"] = ok_count
        df.loc[df["Name"] == name, "final_images_not"] = not_count

# Save the updated DataFrame to the Excel file
df.to_excel(excel_path, index=False)
