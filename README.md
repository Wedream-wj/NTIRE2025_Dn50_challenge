# [NTIRE 2025 Challenge on Image Denoising](https://cvlai.net/ntire/2025/) @ [CVPR 2025](https://cvpr.thecvf.com/)

This is an example of adding noise and simple baseline model.

## How to add noise to images?
`
python add_noise.py
`

## How to test the NRDenosing model?

1. 下载github代码

```bash
git clone https://github.com/Wedream-wj/NTIRE2025_Dn50_challenge.git
```

2. 使用NAFNet模型进行推理；如果觉得推理慢，可直接下载[谷歌云盘](https://drive.google.com/drive/folders/1-uL5N8Ff0CINCq8Lhxjx5WQUZyfuOEz2?usp=sharing)上的06_NAFNetLocal.zip解压放置到NTIRE2025_Challenge/results目录下

```bash
CUDA_VISIBLE_DEVICES=0 python test_demo.py --model_id 6
```

3. 使用RCAN模型进行推理，取消test_demo.py第55到66行的注释；如果觉得推理慢，可直接下载[谷歌云盘](https://drive.google.com/drive/folders/1-uL5N8Ff0CINCq8Lhxjx5WQUZyfuOEz2?usp=sharing)上的06_RCAN.zip解压放置到NTIRE2025_Challenge/results目录下

```bash
CUDA_VISIBLE_DEVICES=0 python test_demo.py --model_id 6
```

4. 进行模型集成，运行ensemble_NRDenosing.py文件；如果觉得慢，可直接下载[谷歌云盘](https://drive.google.com/drive/folders/1-uL5N8Ff0CINCq8Lhxjx5WQUZyfuOEz2?usp=sharing)上的06_NRDenosing_ensemble.zip解压放置到NTIRE2025_Challenge/results目录下

```bash
python ensemble_NRDenosing.py
```

5. `NTIRE2025_Challenge/results/06_NRDenosing_ensemble/test`目录下即为最终提交结果
