<div align="center">

<h1> 🌊 Accelerating Streaming Video Large Language Models via Hierarchical Token Compression 🚀 </h1>

<h4 align="center">
  Yiyu Wang<sup>1*</sup>, Xuyang Liu<sup>1,2*†</sup>, Xiyan Gui<sup>1,3</sup>, Xinying Lin<sup>4</sup>, Boxue Yang<sup>1</sup>,
  <br>
  Chenfei Liao<sup>1,5</sup>, Tailai Chen<sup>1</sup>, Linfeng Zhang<sup>1✉</sup>
  <br><br>
  <sup>1</sup> EPIC Lab, Shanghai Jiao Tong University &emsp; <sup>2</sup> Sichuan University
  <br>
  <sup>3</sup> Huazhong University of Science and Technology &emsp; <sup>4</sup> Sun Yat-sen University
  <br>
  <sup>5</sup> Hong Kong University of Science and Technology (Guangzhou)
</h4>

</div>

---

## 🔥 News

* **`2025.12.02`** 🤗🤗 We release our latest work [STC](https://arxiv.org/pdf/2512.00891), **the first** plug-and-play inference acceleration framework for streaming video understanding! [Code](https://github.com/lern-to-write/STC) is available!
* **`2025.08.21`** 🎉🎉 Our [VidCom<sup>2</sup>](https://arxiv.org/abs/2505.14454) has been accepted by **EMNLP 2025** main conference!
* **`2025.05.21`** 🤗🤗 We release [VidCom<sup>2</sup>](https://arxiv.org/abs/2505.14454), a plug-and-play inference acceleration method of **VideoLLMs**. [Code](https://github.com/xuyang-liu16/VidCom2) is available!

## 📌 Highlights

STC is the first token compression framework for plug-and-play acceleration for streaming video understanding:

* **⚡ Streaming-First Design:** Optimized for latency-sensitive applications (e.g., live sports, AR glasses) where frames arrive continuously.
* **🧩 STC-Cacher :** Exploits temporal redundancy by caching visual features for similar frames (Cosine Similarity $> 0.85$), significantly reducing ViT encoding overhead.
* **✂️ STC-Pruner:** Compresses visual tokens *after* encoding to shorten the LLM prefill sequence while preserving spatiotemporal saliency.
* **🔌 Plug-and-Play:** Seamlessly integrates with SOTA VideoLLMs like **ReKV**, **Dispider**, **StreamForest**, and **Livecc**.

## 🦁 Core Codes

**Core Implementation:**
* **Cache Logic:** [`model/cache.py`](model/cache.py) (Class: `STC_CACHE`)
* **Prune Logic:** [`model/prune.py`](model/prune.py) (Class: `STC_Pruner`)

## 🛠 Preparation

We support the following models enhanced with STC. Code is coming soon.

| Model Base | Status | Code Path |
| :--- | :--- | :--- |
| **ReKV (LLaVA-OV)** | ✅ Supported | [`model/llava_onevision_rekv.py`](model/llava_onevision_rekv.py) |
| **StreamForest** | 🚧 Coming Soon | - |
| **Dispider** | 🚧 Coming Soon | - |
| **LiveCC** | 🚧 Coming Soon | - |

### Environment Settings
#### Original Models (recommended)

We evaluated our model under the same environments as the original models.
So you may set the environments through following the requirements of the mentioned original models.

Links:

| Original  Models |                     urls                     |
| :--------------: | :------------------------------------------: |
|       ReKV        |   https://github.com/Becomebright/ReKV    |
|     StreamForest  | https://github.com/MCG-NJU/StreamForest |
|     Dispider     |    https://github.com/Mark12Ding/Dispider    |
|      LiveCC       |  https://github.com/showlab/livecc   |


Besides, we provide a replica for our environment here:

<details>
<summary>From our environment.yaml</summary>

##### ReKV

  ```bash
  cd ReKV
  conda env create -f environment-ReKV.yml
  ```

##### StreamForest

  ```bash
  cd StreamForest
  conda env create -f environment-StreamForest.yml
  ```

##### Dispider

  ```bash
  cd Dispider
  conda env create -f environment-Dispider.yml
  pip install -v . # for development mode, `pip install -v -e .`
  ```

##### LiveCC

  ```bash
  cd LiveCC
  conda env create -f environment-LiveCC.yml
  pip install -v . # for development mode, `pip install -v -e .`
  ```
</details>

## 🚀 Performance Evaluation

We evaluate STC on both **Online (Streaming)** benchmarks to demonstrate real-time capabilities and **Offline** benchmarks to ensure robust general video understanding.

### 🌊 Online Benchmarks (Streaming)

These benchmarks evaluate the model's ability to understand videos in a streaming fashion, where frames are received sequentially.

#### 1\. StreamingBench

Download the dataset from [mjuicem/StreamingBench](https://huggingface.co/datasets/mjuicem/StreamingBench).

  * **Required files:** `Real_Time_Visual_Understanding.csv` and `Real-Time Visual Understanding_*.zip`.

#### 2\. OVO-Bench

  * **Videos:** Download `src_videos.tar.parta[a-e]` from [JoeLeelyf/OVO-Bench (HF)](https://huggingface.co/datasets/JoeLeelyf/OVO-Bench).
  * **Metadata:** Download `ovo_bench_new.json` from [JoeLeelyf/OVO-Bench (Github)](https://github.com/JoeLeelyf/OVO-Bench).

-----

### 💾 Offline Benchmarks (Standard)

**Supported Datasets:** `MLVU`, `EgoSchema`, `Videomme`

We use standard benchmarks to verify that STC maintains high performance on general video understanding tasks.

- Download benchmarks under `data/`
  - [MLVU-dev](https://huggingface.co/datasets/MLVU/MVLU)
  - [EgoSchema](https://huggingface.co/datasets/lmms-lab/egoschema)
  - [VideoMME](https://huggingface.co/datasets/lmms-lab/Video-MME)

### Run ReKV
#### `MLVU`, `EgoSchema`, `Videomme`
```bash
# Example: Evaluating on MLVU
python video_qa/run_eval.py \
    --model llava_ov_7b \
    --dataset mlvu \
    --num_chunks 8 \
    --sample_fps 0.5 \
    --retrieve_size 64
```
To evaluate `egoschema` or `videomme`, simply change the `--dataset` argument to the respective dataset name.

#### `OVO-Bench`
* **Configuration:** Update `eval/scripts/eval_ovobench.sh`:
    * Set `TASK_JSON` to the path of `ovo_bench_new.json`.
    * Set `VIDEO_DIR` to the unzipped video directory.
```bash 
bash online_bench_inference/OVO-Bench/scripts/inference/rekv.sh

```

#### `StreamingBench`
* **Configuration:** Update `eval/scripts/eval_streamingbench.sh`:
    * Set `TASK_CSV` to the path of the CSV file.
    * Set `VIDEO_DIR` to the unzipped video directory.
```bash
bash online_bench_inference/StreamingBench/scripts/eval_rekv.sh

```


### Run StreamForest
#### `MLVU`, `EgoSchema`, `Videomme`,`OVO-Bench`,`StreamingBench`
```bash
TODO
```
### Run Dispider
#### `OVO-Bench`
```bash
TODO
```
### Run LiveCC
#### `OVO-Bench`
```bash
TODO
```

## 👍 Acknowledgment

We thank the open-source efforts of [LLaVA-OneVision](https://github.com/LLaVA-VL/LLaVA-NeXT) and [ReKV](https://github.com/Becomebright/ReKV).


- Thanks to [ReKV](https://github.com/Becomebright/ReKV) for their great work and codebase.
- Thanks to [StreamForest](https://github.com/MCG-NJU/StreamForest) for their great work and codebase.
- Thanks to [Dispider](https://github.com/Mark12Ding/Dispider) for their great work and codebase.
- Thanks to [LiveCC](https://github.com/showlab/livecc) for their great work and codebase.


## ✏️ Citation


Please consider citing our paper in your publications, if our findings help your research.


```bibtex
@misc{wang2025acceleratingstreamingvideolarge,
      title={Accelerating Streaming Video Large Language Models via Hierarchical Token Compression}, 
      author={Yiyu Wang and Xuyang Liu and Xiyan Gui and Xinying Lin and Boxue Yang and Chenfei Liao and Tailai Chen and Linfeng Zhang},
      year={2025},
      eprint={2512.00891},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2512.00891}, 
}
```
## 📩 Contact

For any questions, please open an issue or contact `ustywan8@ljmu.ac.uk`.
