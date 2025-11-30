import os
import shutil
import argparse

def main(src, dest):
    # Ensure the target folder exists
    os.makedirs(dest, exist_ok=True)

    # Traverse the source folder and its subfolders
    for root, dirs, files in os.walk(src):
        if '__MACOSX' in root:
            print(f"Skipping: {root}")
            continue

        for file in files:
            if file.endswith('.mp4'):
                TYPE = root.split('/')[-2].replace(' ', '_')
                sample_folder = os.path.basename(root)
                new_file_name = f"{sample_folder}_{TYPE}.mp4"
                src_file_path = os.path.join(root, file)
                dest_file_path = os.path.join(dest, new_file_name)
                
                # Move and rename the file
                shutil.move(src_file_path, dest_file_path)
                print(f"Moved: {src_file_path} to {dest_file_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', type=str, required=True, help='Source folder path')
    parser.add_argument('--dest', type=str, required=True, help='Destination folder path')
    args = parser.parse_args()
    main(args.src, args.dest)