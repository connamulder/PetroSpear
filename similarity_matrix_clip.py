import torch
from transformers import CLIPProcessor, CLIPModel
import torch.nn.functional as F

from dataset_rocks_labels_name import get_rock_labels_name


def build_similarity_matrix_by_clip(dataset_name):
    # 加载模型和处理器
    model_name = "openai/clip-vit-base-patch16"
    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name)
    model.eval()

    # 根据岩石图像数据集获取准备岩石英文名称列表
    rock_names_en, _ = get_rock_labels_name(dataset_name)

    # 文本编码 (Tokenize)
    inputs = processor(text=rock_names_en, return_tensors="pt", padding=True, truncation=True)

    # 提取文本特征向量
    with torch.no_grad():
        text_features = model.get_text_features(**inputs)

    # L2 归一化与计算余弦相似度
    text_features = F.normalize(text_features, p=2, dim=-1)
    cosine_sim_matrix = torch.matmul(text_features, text_features.T).numpy()

    return cosine_sim_matrix


def main():
    dataset_name = "plutonicrocks13_v3"
    rock_names_en, _ = get_rock_labels_name(dataset_name)
    sim_matrix = build_similarity_matrix_by_clip(dataset_name)

    from utils.utils_draw_confusion_matrix import draw_sim_matrix

    draw_sim_matrix(sim_matrix, rock_names_en, "sim_matrix_clip", "output")

    """
    # 打印结果
    df = pd.DataFrame(sim_matrix, index=rock_names_en, columns=rock_names_en)
    print("=== 岩石英文名称余弦相似度矩阵 ===")
    print(df.round(4))

    tensor = torch.from_numpy(sim_matrix).float()
    print(tensor)
    """


if __name__ == '__main__':
    main()