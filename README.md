# RGBD_Video_SOD
# [IJCV-2024] ViDSOD-100: A New Dataset and a Baseline Model for RGB-D Video Salient Object Detection
This repository is the official implementation of [ViDSOD-100](https://link.springer.com/article/10.1007/s11263-024-02051-5#Abs1)


## TODO List
- [x]  ~~release dataset~~
- [x] ~~release code~~
- [x] ~~release weights~~


## Data
You can download our ViDSOD-100 dataset from [here](https://drive.google.com/file/d/1UDPHdgygVJxuAigJuBy8aTPRt8A6Our9/view?usp=sharing).<br>
For both training and inference, the following dataset structure is required:

```
vidsod_100
|-- train
|-- train_flow
|-- test
|-- test_flow
```

## training
``` shell
sh train.sh
```


## inference
``` shell
sh infer.sh
```

## evaluation
``` shell
sh cal_scores.sh
```

## Weights
We released our [checkpoint](https://drive.google.com/file/d/1WNvwL6_ZxAX6oaDUmAjpE17H_xOV6tET/view?usp=sharing) on ViDSOD-100 <br>

## Saliency Maps


## Citation
If you find our work useful for your research, please cite us:
```
@article{lin2024vidsod,
  title={ViDSOD-100: A New Dataset and a Baseline Model for RGB-D Video Salient Object Detection},
  author={Lin, Junhao and Zhu, Lei and Shen, Jiaxing and Fu, Huazhu and Zhang, Qing and Wang, Liansheng},
  journal={International Journal of Computer Vision},
  pages={1--19},
  year={2024},
  publisher={Springer}
}
```

