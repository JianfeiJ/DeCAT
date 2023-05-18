# DWT-MVSNet
Code will be release soon.

## Dataset
We use [DTU](http://roboimagedata.compute.dtu.dk/?page_id=36) and [BlendedMVS](https://github.com/YoYo000/BlendedMVS) dataset for training, and [DTU](http://roboimagedata.compute.dtu.dk/?page_id=36), [Tanks & Temples](https://www.tanksandtemples.org/), and [ETH3D](https://www.eth3d.net/) for testing and evaluation.  
To download pre-processed datasets, please follow the instructions provided on [PatchmatchNet](https://github.com/FangjinhuaWang/PatchmatchNet).

## Evaluation Performance
### DTU
|    Methods  |  Acc. (mm)     | Comp. (mm) | Overall (mm)   |
|    :----:   |    :----:   |    :----:   |    :----:    |
| PatchmatchNet(1600×1152)      | 0.427      |0.277| 0.352   |
| Ours(1152×864)      | 0.442      |0.262| 0.352   |
| Ours(1600×1152)      | 0.427      |0.256| 0.342   |
### Tanks & Temples (F-score)
|    Training Dataset  |Intermediate|Advanced |
|    :----:   |    :----:   |    :----:   |
| DTU      | 56.70|35.59|
| BlendedMVS      | 57.67      |35.89|

### ETH3D (F-score)
|  Training  |  Test  |
|    :----:   |    :----:   |
|67.24|73.70|

## Point cloud visualizations
<img src="https://github.com/JianfeiJ/DWT-MVSNet/blob/main/images/DTU_Compare.png">
