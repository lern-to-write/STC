from huggingface_hub import snapshot_download
import os

def check_and_create_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
        
download_worknum = 1


download_id = "MCG-NJU/StreamForest-Annodata"
download_type= "dataset"
download_dir = "./anno"

if __name__ == "__main__":
    check_and_create_dir(download_dir)
    snapshot_download(repo_type= download_type, repo_id = download_id, local_dir = download_dir, max_workers=download_worknum)


download_id = "MCG-NJU/StreamForest-Qwen2-7B"
download_type= "model"
download_dir = "./ckpt/stage4-postft-qwen-siglip/StreamForest-Qwen2-7B"

if __name__ == "__main__":
    check_and_create_dir(download_dir)
    snapshot_download(repo_type= download_type, repo_id = download_id, local_dir = download_dir, max_workers=download_worknum)
    

download_id = "MCG-NJU/StreamForest-Drive-Qwen2-7B"
download_type= "model"
download_dir = "./ckpt/stage5-driveft-qwen-siglip/StreamForest-Drive-Qwen2-7B"

if __name__ == "__main__":
    check_and_create_dir(download_dir)
    snapshot_download(repo_type= download_type, repo_id = download_id, local_dir = download_dir, max_workers=download_worknum)
    

download_id = "MCG-NJU/StreamForest-Pretrain-Qwen2-7B"
download_type= "model"
download_dir = "./ckpt/stage3-video_sft/StreamForest-Pretrain-Qwen2-7B"

if __name__ == "__main__":
    check_and_create_dir(download_dir)
    snapshot_download(repo_type= download_type, repo_id = download_id, local_dir = download_dir, max_workers=download_worknum)