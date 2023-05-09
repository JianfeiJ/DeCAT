# DWT-MVSNet
Code will be release soon.

## Dataset
We use DTU and BlendedMVS dataset for training, and DTU, Tanks & Temples, and ETH3D for testing and evaluation. Please follow [IterMVS](https://github.com/FangjinhuaWang/IterMVS).

## Evaluation Performance
### DTU
| Acc. (mm)| Comp. (mm)| Overall (mm)|
|    :----:   |    :----:   |     :----:   | 
| 0.427      |**0.256**| 0.342   |
### Tanks & Temples-trained on DTU (F-score)
|Intermediate|Advanced|
|    :----:   |    :----:   |
|56.70|35.59|

### Tanks & Temples-trained on BlendedMVS (F-score)
|Intermediate|Advanced|
|    :----:   |    :----:   |
|57.67|35.89|

### ETH3D-trained on BlendedMVS
|  Training  |  Test  |
|    :----:   |    :----:   |
|  67.24  |  73.70  |

## Some point cloud visualizations
<img src="https://github.com/JianfeiJ/DWT-MVSNet/blob/main/images/DWT-MVSNet_scan24.png" width="50%">
<img src="https://github.com/JianfeiJ/DWT-MVSNet/blob/main/images/Horse.png" width="50%">
