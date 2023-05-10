# DWT-MVSNet
Code will be release soon.

## Dataset
We use [DTU](http://roboimagedata.compute.dtu.dk/?page_id=36) and [BlendedMVS](https://github.com/YoYo000/BlendedMVS) dataset for training, and [DTU](http://roboimagedata.compute.dtu.dk/?page_id=36), [Tanks & Temples](https://www.tanksandtemples.org/), and [ETH3D](https://www.eth3d.net/) for testing and evaluation. To access more information about downloading pre-processed datasets, please follow the instructions provided on [IterMVS](https://github.com/FangjinhuaWang/IterMVS).

## Evaluation Performance
### DTU
| Acc. (mm)| Comp. (mm)| Overall (mm)|
|    :----:   |    :----:   |     :----:   | 
| 0.427      |**0.256**| 0.342   |
### Tanks & Temples-trained on DTU (F-score)
|Intermediate|Advanced|
|    :----:   |    :----:   |
|56.70|**35.59**|

### Tanks & Temples-trained on BlendedMVS (F-score)
|Intermediate|Advanced|
|    :----:   |    :----:   |
|57.67|**35.89**|

### ETH3D-trained on BlendedMVS
|  Training  |  Test  |
|    :----:   |    :----:   |
|  67.24  |  73.70  |

## Point cloud visualizations
<img src="https://github.com/JianfeiJ/DWT-MVSNet/blob/main/images/DTU_Compare.png">
