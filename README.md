# TBA

### Download the dataset

Download the dataset from TBA.

### Calibrating cameras (optional)

To recreate the calibration data (`eval_data/calib_data.npy`) run the following. 

```sh
python eval_utils/calibrate_stereo.py path/to/dataset/calibration
```

### Setting up Gaze Estimation Networks

#### GazeTR
* Download weights from [google drive](https://drive.google.com/file/d/1WEiKZ8Ga0foNmxM7xFabI4D5ajThWAWj/view?usp=sharing) or [baidu cloud disk](https://pan.baidu.com/s/1GEbjbNgXvVkisVWGtTJm7g) with code 1234 and save to `GazeTR_net/weights`.

#### L2CS
* Download weights from [google drive](https://drive.google.com/drive/folders/17p6ORr-JQJcw-eYtG2WGNiuS_qVKwdWd?usp=sharing) and place them in `L2CS/weights`.

#### 3DGazenet
* Download [data folder with weights](https://drive.google.com/file/d/1mYvKRJGS8LY5IU3I8Qfvm-xINQyby1z5/view?usp=sharing) and unzip into `GazeNet/data`.

#### Gaze3D
* Download [face detector weights](https://drive.usercontent.google.com/download?id=1gglIwqxaH2iTvy6lZlXuAcMpd_U0GCUb&export=download&authuser=0) and place them in `gaze3d/weights`.
* Download [gaze prediction weights](https://github.com/idiap/gaze3d/raw/refs/heads/main/checkpoints/gat_stwsge_gaze360_gf.ckpt) and place them in `gaze3d/weights`.


### Run the evaluation

```shell
python evaluate.py path/to/downloaded_dataset/eyetracker
```