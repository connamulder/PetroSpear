import os.path

from py2neo import Graph, RelationshipMatcher, NodeMatcher
import numpy as np

from dataset_rocks_labels_name import get_rock_labels_name


def build_similarity_matrix_by_neo4j(dataset_name):
    # 根据岩石图像数据集获取准备岩石中文名称列表
    _, id2label = get_rock_labels_name(dataset_name)
    class_num = len(id2label)
    print(id2label)

    sim_matrix = np.zeros((class_num, class_num))

    matrix_file_name = 'sim_matrix_%s.npy' % dataset_name
    matrix_file_name_dir = 'data/%s' % matrix_file_name

    if os.path.exists(matrix_file_name_dir):
        sim_matrix = np.load(matrix_file_name_dir)
    else:
        try:
            # graph = Graph("bolt://localhost:7687", user="neo4j", password="Neo4j861224", name="neo4j")
            graph = Graph('bolt://localhost:7687/', auth=("neo4j", "Neo4j20240624"))
            node_matcher = NodeMatcher(graph)
            relation_matcher = RelationshipMatcher(graph)
            print("Neo4j服务器连接成功！")
        except BaseException:
            print("Neo4j服务器连接失败！")


        subject_type = 'Rock'
        object_type = 'Rock'
        predicate_type = 'SIMILAR'
        for i in range(class_num):
            subject_name = id2label[i]
            for j in range(class_num):
                if i == j:
                    sim_matrix[i][j] = 1.0
                else:
                    object_name = id2label[j]
                    subject_node = node_matcher.match(subject_type).where(name=subject_name).first()
                    object_node = node_matcher.match(object_type).where(name=object_name).first()
                    if subject_node is not None and object_node is not None:
                        relation_score = list(relation_matcher.match((subject_node, object_node), r_type=predicate_type))
                        if len(relation_score) > 0:
                            score_value = relation_score[0]['score']
                            sim_matrix[i][j] = score_value
        # 保存为 .npy 文件（二进制格式，高效）
        np.save(matrix_file_name_dir, sim_matrix)


    return sim_matrix


def main():
    dataset_name = "plutonicrocks13_v3"
    rock_names_en, _ = get_rock_labels_name(dataset_name)
    sim_matrix = build_similarity_matrix_by_neo4j(dataset_name)

    from utils.utils_draw_confusion_matrix import draw_sim_matrix

    draw_sim_matrix(sim_matrix, rock_names_en, "sim_matrix_node", "output")

    """
    # 打印结果
    df = pd.DataFrame(sim_matrix, index=rock_names_en, columns=rock_names_en)
    print("=== 岩石余弦相似度矩阵 ===")
    print(df.round(6))

    tensor = torch.from_numpy(sim_matrix).float()
    print(tensor)
    """


if __name__ == '__main__':
    main()