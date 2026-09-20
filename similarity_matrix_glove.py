"""
    @Project: PetroSpear
    @File   : similarity_matrix_glove.py
    @Author : mulder
    @E-mail : c_mulder@163.com
    @Date   : 2026-09-03
    @Info   : To get rock type word embedding vectors of the the geoscience GloVe model.
"""

import os
import numpy as np
from dataset_rocks_labels_name import get_rock_labels_name

import torch
import torch.nn.functional as F


# 全局变量，仅加载一次
word_to_vec = {}


def get_glove_vector(file_path, target_words):
    """
    从 GloVe 格式的文件中加载并查找目标岩石类型（单词）的向量
    """
    if not os.path.exists(file_path):
        print(f"未找到文件: {file_path}")
        return None

    if len(word_to_vec) == 0:
        print(f"正在加载 {file_path}，可能需要十多秒钟...")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    values = line.split()
                    word = values[0]
                    vector = np.asarray(values[1:], dtype='float32')
                    word_to_vec[word] = vector
            print(f"加载完成，共收录 {len(word_to_vec)} 个单词。")
        except Exception as e:
            print(f"读取文件出错: {e}")
            return None

    # 查找目标岩石类型 (统一转为小写)
    target_words = target_words.lower().split()
    word_vectors = []
    for target_word in target_words:
        if target_word in word_to_vec:
            vec = word_to_vec[target_word]
            word_vectors.append(vec)
    vector = np.mean(word_vectors, axis=0)

    return vector


def get_text_features(rock_names):
    class_num = len(rock_names)
    vector_dim = 300

    glove_vectors_file_name = 'glove_vectors.txt'
    glove_vectors_file_path = 'data/%s' % glove_vectors_file_name

    texts_features = np.zeros((class_num, vector_dim), dtype=np.float32)

    for iter in range(class_num):
        rock_name = rock_names[iter]
        vector_temp = get_glove_vector(glove_vectors_file_path, rock_name)
        texts_features[iter] = vector_temp

    return texts_features


def build_similarity_matrix_by_glove(dataset_name):
    matrix_file_name = 'sim_matrix_%s_glove.npy' % dataset_name
    matrix_file_name_dir = 'data/%s' % matrix_file_name

    if os.path.exists(matrix_file_name_dir):
        cosine_sim_matrix = np.load(matrix_file_name_dir)
    else:
        # 根据岩石图像数据集获取准备岩石英文名称列表
        rock_names_en, _ = get_rock_labels_name(dataset_name)
        text_features = get_text_features(rock_names_en)
        text_features = torch.from_numpy(text_features)

        # L2 归一化与计算余弦相似度
        text_features = F.normalize(text_features, p=2, dim=-1)
        cosine_sim_matrix = torch.matmul(text_features, text_features.T).numpy()

        # 保存为 .npy 文件（二进制格式，高效）
        np.save(matrix_file_name_dir, cosine_sim_matrix)

    return cosine_sim_matrix


def main():
    """
    rock_type = 'olivine'

    glove_vectors_file_name = 'vectors.txt'
    glove_vectors_file_path = 'data/%s' % glove_vectors_file_name

    vector = get_glove_vector(glove_vectors_file_path, rock_type)
    print("%s: %s" % (rock_type, vector))

    rock_type = 'nepheline syenite'
    vector = get_glove_vector(glove_vectors_file_path, rock_type)
    print("%s: %s" % (rock_type, vector))
    """

    dataset_name = "plutonicrocks13_v3_Iter3"
    rock_names_en, _ = get_rock_labels_name(dataset_name)
    sim_matrix = build_similarity_matrix_by_glove(dataset_name)

    from utils.utils_draw_confusion_matrix import draw_sim_matrix

    filename = 'sim_matrix_%s_glove' % dataset_name
    draw_sim_matrix(sim_matrix, rock_names_en, filename, "output")


if __name__ == '__main__':
    main()