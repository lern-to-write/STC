import os
import json
import argparse

def update_video_path_to_absolute(json_file):
    with open(json_file, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    if "sqa" in json_file:
        for subset in data:
            for entry in subset:
                if 'video_path' in entry:
                    entry['video_path'] = os.path.abspath(entry['video_path'])
    else:
        for entry in data:
            if 'video_path' in entry:
                entry['video_path'] = os.path.abspath(entry['video_path'])
    
    with open(json_file, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=4, ensure_ascii=False)

def main():
    parser = argparse.ArgumentParser(description="Convert relative video paths in a JSON file to absolute paths.")
    parser.add_argument("--src", type=str, help="Path to the src JSON file.")
    args = parser.parse_args()
    
    update_video_path_to_absolute(args.src)

if __name__ == "__main__":
    main()