# PetroSpear: Data- and kowledge-driven petrographic image classification with semantic prior embedding and ranking loss

<div align="center">
<strong>Author: Zhongliang Chen, Chaojie Zheng, Xiaohui Li, Feng Yuan</strong>
  
<strong>Geological Survey of Anhui Province (Anhui Institute of Geological Sciences)</strong>
</div>

This is the official repository for paper **"PetroSpear: Data- and kowledge-driven petrographic image classification with semantic prior embedding and ranking loss"**. [[paper](https://)]

## Please share a <font color='orange'>STAR ⭐</font> if this project does help


## Preparation
Create a virtual environment and install the required libraries.
```shell
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/
```

## Running the code

### 1. PetroMind Instruction Fine-tuning
Execute the PetroMind instruction fine-tuning script.
```shell
01_train_two_stage.sh
```

### 2. CNNs Training on the Rocks-13
Execute the CNNs training script for the Rocks-13 dataset.
```shell
02_rocks13_cnn.sh
```


```bash
@article{chen2026petrospear,
  title={PetroSpear: Data- and kowledge-driven petrographic image classification with semantic prior embedding and ranking loss},
  author={Chen Zhongliang,Zheng Chaojie, Li Xiaohui, Yuan Feng},
  journal={},
  year={},
  publisher={},
  doi={}
}
```

## 🙏 Acknowledgement
We thank Cochise College (USA) for granting permission to use the rock images from their website (https://geology.cochise.edu) free of charge for non-commercial educational purposes. We also extend our gratitude to Grahan Wilson / Turnstone Geological Services for sharing and granting permission to use the graphic granite images from their website (https://turnstone.ca).

## 🤖 Contributing
Feel free to contact c_mulder@163.com if you have any question.
