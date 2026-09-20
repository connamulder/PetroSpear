# PetroSpear: Data- and kowledge-driven petrographic image classification with semantic prior embedding and ranking loss

<div align="center">
<strong>Author: Zhongliang Chen<sup>a</sup>, Chaojie Zheng<sup>b, c</sup>, Xiaohui Li<sup>b, c</sup> and Feng Yuan<sup>b, c</sup></strong>
  
<strong><sup>a</sup> Geological Survey of Anhui Province (Anhui Institute of Geological Science), Hefei 230001, China </strong><br>
<strong><sup>b</sup> School of Resources and Environment Engineering, Hefei University of Technology, Hefei 230009, China </strong><br>
<strong><sup>c</sup> Anhui Provincial Key Laboratory for Deep Exploration, Evaluation and Utilization of Strategic Mineral Resources, Hefei 230009, China.</strong><br>
</div>

This is the official repository for paper **"PetroSpear: Data- and kowledge-driven petrographic image classification with semantic prior embedding and ranking loss"**. [[paper](https://)]

## Please share a <font color='orange'>STAR ⭐</font> if this project does help


## Preparation
Create a virtual environment.
```shell
conda create --name torch_spear python=3.11.11
conda activate torch_spear
```

Install torch-gpu, torchvision, and torchaudio, and then install the other dependencies.
```shell
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/
```

## Running the code

### 1. PetroSpear two-stage training
Run the PetroSpear two-stage training script using pre-trained VGG16, ResNet50, and ViT-Base-Patch16-224 backbones.
```shell
./01_cnn_train_multi_loss_rank.sh
```

### 2. Sensitivity analysis of early-stopping patience
Run repeated comparative experiments using ResNet50 with different early stopping patience settings.
```shell
./02_cnn_train_multi_loss_rank_patient.sh
./03_cnn_train_multi_loss_rank_iter.sh
```

### 3. Impact of loss function weights
Run the sensitivity analysis script using varying Spear loss weight settings.
```shell
./04_cnn_train_multi_loss_rank_weight.sh
```

### 4. Analysis of semantic similarity matrix calculation methods
Run the script of comparative experiments on different approaches for semantic similarity matrix computation.
```shell
./05_cnn_train_multi_loss_rank_emb.sh
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
