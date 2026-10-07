# Gaze Estimation for Human-Robot Interaction: Analysis Using the NICO Platform

Pre-print: TBA.

Abstract: This paper evaluates the current gaze estimation methods within a human-robot interaction (HRI) context of a shared workspace scenario. We introduce a new, annotated dataset collected with the NICO robotic platform. We evaluate four state-of-the-art gaze estimation models. The evaluation shows that the angular errors are close to those reported on general-purpose benchmarks. However, when expressed in terms of distance in the shared workspace the best median error is 14.57 cm, quantifying the practical limitations of current methods. We conclude by discussing these limitations and offering recommendations on how to best integrate gaze estimation as a modality in HRI systems.

## Running the code

### Download the dataset

Download the dataset from [Zenodo](https://doi.org/10.5281/zenodo.23059771).

### Calibrating cameras (optional)

To recreate the calibration data (`eval_data/calib_data.npy`) run the following. 

```sh
python eval_utils/calibrate_stereo.py path/to/dataset/calibration
```

The calibration pattern was shown on a display, so its squares were not the nominal 32 mm. The metric scale is therefore fixed by rescaling the calibration so that the stereo baseline matches the measured 70 mm (`--baseline`).

### Setting up Gaze Estimation Networks

#### GazeTR
* Download weights from [google drive](https://drive.google.com/file/d/1WEiKZ8Ga0foNmxM7xFabI4D5ajThWAWj/view?usp=sharing) or [baidu cloud disk](https://pan.baidu.com/s/1GEbjbNgXvVkisVWGtTJm7g) with code 1234 and save to `GazeTR_net/weights`.

#### L2CS
* Download weights from [google drive](https://drive.google.com/drive/folders/17p6ORr-JQJcw-eYtG2WGNiuS_qVKwdWd?usp=sharing) and place them in `L2CS/weights`.

#### 3DGazenet
* Download [data folder with weights](https://drive.google.com/file/d/1mYvKRJGS8LY5IU3I8Qfvm-xINQyby1z5/view?usp=sharing) and unzip into `GazeNet/data`.

#### Gaze3D
* Download [face detector weights](https://drive.usercontent.google.com/download?id=1gglIwqxaH2iTvy6lZlXuAcMpd_U0GCUb&export=download&authuser=0) and place them in `gaze3d/checkpoints`.
* Download [gaze prediction weights](https://github.com/idiap/gaze3d/raw/refs/heads/main/checkpoints/gat_stwsge_gaze360_gf.ckpt) and place them in `gaze3d/checkpoints`.


### Run the evaluation

```shell
python eval.py path/to/downloaded_dataset/eyetracker
```
## Acknowledgment

We have used code from the following repositories:

* GazeTR: https://github.com/yihuacheng/GazeTR
* L2CS: https://github.com/Ahmednull/L2CS-Net
* 3DGazeNet: https://github.com/eververas/3DGazeNet
* Gaze3D https://github.com/idiap/gaze3d

If you use the code please cite also the original publications:

```
@inproceedings{cheng2022GazeTR,
  title={Gaze estimation using transformer},
  author={Cheng, Yihua and Lu, Feng},
  booktitle={2022 26th International Conference on Pattern Recognition (ICPR)},
  pages={3341--3347},
  year={2022},
  organization={IEEE}
}

@inproceedings{abdelrahman2023L2CS,
  title={L2cs-net: Fine-grained gaze estimation in unconstrained environments},
  author={Abdelrahman, Ahmed A and Hempel, Thorsten and Khalifa, Aly and Al-Hamadi, Ayoub and Dinges, Laslo},
  booktitle={2023 8th International Conference on Frontiers of Signal Processing (ICFSP)},
  pages={98--102},
  year={2023},
  organization={IEEE}
}

@inproceedings{ververas20243dGazeNet,
  title={3DGazeNet: Generalizing 3D gaze estimation with weak-supervision from synthetic views},
  author={Ververas, Evangelos and Gkagkos, Polydefkis and Deng, Jiankang and Doukas, Michail Christos and Guo, Jia and Zafeiriou, Stefanos},
  booktitle={European Conference on Computer Vision},
  pages={387--404},
  year={2024},
  organization={Springer}
}

@inproceedings{vuillecard2025Gaze3D,
  author = {Vuillecard, Pierre and Odobez, Jean-Marc},
  month = jun,
  title = {Enhancing 3D Gaze Estimation in the Wild using Weak Supervision with Gaze Following Labels},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2025},
}
```