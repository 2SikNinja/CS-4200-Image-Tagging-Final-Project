import os


def rename_files(folder_path, prefix):
    files = os.listdir(folder_path)
    for index, file in enumerate(files):
        # Get the file extension
        _, extension = os.path.splitext(file)

        # Construct the new file name
        new_name = f"{prefix}{index + 1}{extension}"

        # Construct the full paths of the old and new names
        old_path = os.path.join(folder_path, file)
        new_path = os.path.join(folder_path, new_name)

        # Rename the file
        os.rename(old_path, new_path)
        print(f"Renamed {file} to {new_name}")


folder_path = 'E:\Documents\School\Spring 2023\CS 4200 Artificial Intelligence\FINALPROJECTFROMSCRATCH\/raw-img\squirrel'
prefix = 'squirrel'

# Call the function to rename the files
rename_files(folder_path, prefix)
