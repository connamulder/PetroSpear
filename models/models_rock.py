"""
    @Project: PetroSpear
    @File   : models/models_rock.py
    @Author : mulder
    @E-mail : c_mulder@163.com
    @Date   : 2026-07-01
    @Info   : 根据输入的 model_type，定义各类岩石图像分类模型。
              CNNs:      ‘vgg16’ | ‘inception_v3’ | ‘resnet50’ | ‘desnet’ | ‘effinet’
              Transformer: ‘vit_base’ | 'mobilevit'
"""

import torch
import json

from torchvision.models import inception_v3, vgg16, resnet50
from torchvision.models import densenet121, DenseNet121_Weights, efficientnet_b7, EfficientNet_B7_Weights
import timm


class RockModel(torch.nn.Module):
    def __init__(self, class_num=10, model_type='vgg16', pretrained=True, img_size=224, labels_name=None):
        super(RockModel, self).__init__()
        self.class_num = class_num
        self.model_type = model_type.lower()
        self.pretrained = pretrained
        self.img_size = img_size
        self.num_feature = 1000

        self.id2label = None
        self.label2id = None
        if labels_name is not None:
            self.id2label = dict(zip(labels_name.keys(), labels_name.values()))
            # print("RockModel id2label: ", self.id2label)
            # print(f"RockModel id2label: {self.id2label}")
            print(f"RockModel id2label:\n{json.dumps(self.id2label, indent=4, ensure_ascii=False)}")
            self.label2id = dict(zip(labels_name.values(), labels_name.keys()))
            # print(self.label2id)
            print(f"RockModel label2id:\n{json.dumps(self.label2id, indent=4, ensure_ascii=False)}")

        if self.model_type == "vgg16":
            if self.pretrained:
                self.model = vgg16(pretrained=True)
            else:
                self.model = vgg16(pretrained=False)
                self.model.classifier[6] = torch.nn.Linear(4096, self.class_num)
        elif self.model_type == "inception_v3":
            if self.pretrained:
                self.model = inception_v3(pretrained=True)
            else:
                self.model = inception_v3(pretrained=False)
                # self.model.fc = torch.nn.Linear(self.model.fc.in_features, self.class_num)
            self.model.aux_logits = False
        elif self.model_type == "resnet50":
            if self.pretrained:
                self.model = resnet50(pretrained=True)
            else:
                self.model = resnet50(pretrained=False)
                num_ftrs = self.model.fc.in_features
                self.model.fc = torch.nn.Linear(num_ftrs, self.class_num)
        elif self.model_type == "desnet":
            if self.pretrained:
                # self.model = densenet121(pretrained=True)
                self.model = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
            else:
                self.model = densenet121(pretrained=False)
                num_ftrs = self.model.classifier.in_features
                self.model.classifier = torch.nn.Linear(num_ftrs, self.class_num)
        elif self.model_type == "effinet":
            if self.pretrained:
                self.model = efficientnet_b7(weights=EfficientNet_B7_Weights.IMAGENET1K_V1)
            else:
                self.model = efficientnet_b7(weights=None)
                num_ftrs = self.model.classifier[1].in_features
                self.model.classifier[1] = torch.nn.Linear(num_ftrs, self.class_num)
        elif self.model_type == "vit_base":
            model_name = "vit_base_patch16_224"
            if self.pretrained:
                self.model = timm.create_model(
                    model_name,
                    pretrained=True,
                    num_classes=self.class_num
                )
        elif self.model_type == "mobilevit":
            model_name = "mobilevit_xs.cvnets_in1k"
            self.num_feature = 384
            if self.pretrained:
                self.model = timm.create_model(model_name,
                                               pretrained=True,
                                               img_size=self.img_size,
                                               num_classes=self.class_num)
            else:
                self.model = timm.create_model(model_name,
                                               pretrained=False,
                                               img_size=self.img_size,
                                               num_classes=self.class_num)

        self.fc = torch.nn.Sequential(
            torch.nn.Linear(self.num_feature, self.class_num),
            torch.nn.Sigmoid(),
        )

    def forward(self, img):
        # 正常训练代码
        if self.model_type == "mobilevit" or self.model_type == "vit_base":
            x = self.model(img)
        else:
            x = self.model(img)
            if self.pretrained:
                x = self.fc(x)

        # Debug模式下：ViT模型pth文件输出为pt时的执行代码
        """
        x = self.model(img)
        x = x.logits
        """

        # Debug模式下：vgg16、resnet等非ViT模型pth文件输出为pt时的执行代码
        """
        x = self.model(img)
        x = self.fc(x)
        """

        return x


if __name__ == '__main__':
    from torchsummary import summary
    from thop import profile

    class_num = 10
    model_type = 'vit_base'
    pretrained = True
    img_size = 224
    model = RockModel(class_num, model_type, pretrained, img_size)
    if model is not None:
        if (model_type == "vit_base") or (model_type == "CLIP") or (model_type == "desnet"):
            print("不输出summary")
        else:
            summary(model, input_size=(3, img_size, img_size), device='cpu')

        batch_size = 16
        input = torch.randn(batch_size, 3, img_size, img_size)  # 假设输入是 1 张 3x224x224 的图片
        flops, params = profile(model, inputs=(input,))

        print("thop模型参数统计:\n")
        print("FLOPs: %.6fG\n" % (flops / (1024 * 1024 * 1024)))
        print("Params: %.6fM\n" % (params / (1024 * 1024)))
    else:
        print("Model is None.")
