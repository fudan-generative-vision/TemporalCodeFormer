<h1 align='center'>Hallo2: Long-Duration and High-Resolution Audio-driven Portrait Image Animation</h1>

## ⚙️ Installation

- System requirement: Ubuntu 20.04/Ubuntu 22.04, Cuda 12.1
- Tested GPUs: A100

Create conda environment:

```bash
  conda create -n hallo python=3.10
  conda activate hallo
```

Install packages with `pip`

```bash
  pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu118
  pip install -r requirements.txt
```

Besides, ffmpeg is also needed:
```bash
  apt-get install ffmpeg
```

### 📥 Download Pretrained Models

You can easily get all pretrained models required by inference from our [HuggingFace repo](https://huggingface.co/fudan-generative-ai/hallo2).

Clone the pretrained models into `${PROJECT_ROOT}/pretrained_models` directory by cmd below:

```shell
git lfs install
git clone https://huggingface.co/fudan-generative-ai/hallo2 pretrained_models
```

Or you can download them separately from their source repo:
- [hallo2](https://huggingface.co/fudan-generative-ai/hallo2/blob/main/hallo2/net_g.pth): Our checkpoint of video super-resolution.
- [facelib](https://github.com/sczhou/CodeFormer/releases/tag/v0.1.0): pretrained face parse models
- [realesrgan](https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/RealESRGAN_x2plus.pth): background upsample model
- [CodeFormer](https://github.com/sczhou/CodeFormer/releases/download/v0.1.0): pretrained [Codeformer](https://github.com/sczhou/CodeFormer) model, it's optional to download it, only if you want to train our video super-resolution model from scratch

Finally, these pretrained models should be organized as follows:

```text
./pretrained_models/
|-- CodeFormer/
|   |-- codeformer.pth
|   `-- vqgan_code1024.pth
|-- facelib
|   |-- detection_mobilenet0.25_Final.pth
|   |-- detection_Resnet50_Final.pth
|   |-- parsing_parsenet.pth
|   |-- yolov5l-face.pth
|   `-- yolov5n-face.pth
|-- hallo2
|   `-- net_g.pth
`-- realesrgan
    `-- RealESRGAN_x2plus.pth
```

### 🎮 Run Inference
#### High-Resolution animation
Simply to run the `scripts/video_sr.py` and pass `input_path` and `output_path`:

```bash
python scripts/video_sr.py --input_path [input_video] --output_path [output_dir] --bg_upsampler realesrgan --face_upsample -w 1 -s 4
```

Animation results will be saved at `output_dir`. 

For more options:

```shell
usage: video_sr.py [-h] [-i INPUT_PATH] [-o OUTPUT_PATH] [-w FIDELITY_WEIGHT] [-s UPSCALE] [--has_aligned] [--only_center_face] [--draw_box]
                   [--detection_model DETECTION_MODEL] [--bg_upsampler BG_UPSAMPLER] [--face_upsample] [--bg_tile BG_TILE] [--suffix SUFFIX]

options:
  -h, --help            show this help message and exit
  -i INPUT_PATH, --input_path INPUT_PATH
                        Input video
  -o OUTPUT_PATH, --output_path OUTPUT_PATH
                        Output folder. 
  -w FIDELITY_WEIGHT, --fidelity_weight FIDELITY_WEIGHT
                        Balance the quality and fidelity. Default: 0.5
  -s UPSCALE, --upscale UPSCALE
                        The final upsampling scale of the image. Default: 2
  --has_aligned         Input are cropped and aligned faces. Default: False
  --only_center_face    Only restore the center face. Default: False
  --draw_box            Draw the bounding box for the detected faces. Default: False
  --detection_model DETECTION_MODEL
                        Face detector. Optional: retinaface_resnet50, retinaface_mobile0.25, YOLOv5l, YOLOv5n. Default: retinaface_resnet50
  --bg_upsampler BG_UPSAMPLER
                        Background upsampler. Optional: realesrgan
  --face_upsample       Face upsampler after enhancement. Default: False
  --bg_tile BG_TILE     Tile size for background sampler. Default: 400
  --suffix SUFFIX       Suffix of the restored faces. Default: None
```

## Training
##### prepare data for training
We use the VFHQ dataset for training, you can download from its [homepage](https://liangbinxie.github.io/projects/vfhq/). Then updata `dataroot_gt` in `./configs/train/video_sr.yaml`.

#### training
Start training with the following command:
```shell
python -m torch.distributed.launch --nproc_per_node=8 --master_port=4652 \
basicsr/train.py -opt ./configs/train/video_sr.yaml \
--launcher pytorch
```
