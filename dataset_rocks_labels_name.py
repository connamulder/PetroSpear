"""
    @Project: PetroSpear
    @File   : dataset_rocks_labels_name.py
    @Author : mulder
    @E-mail : c_mulder@163.com
    @Date   : 2026-07-01
    @Info   : 设定岩石图像数据集英文名称列表和对应的中文名称字典。
"""


class_labels_plutonic_v3 = ['olivinite', 'pyroxene', 'hornblendite', 'gabbro', 'diorite', 'syenite',
                            'monzonite', 'syenogranite', 'monzonitic granite', 'granodiorite', 'plagioclase granite',
                            'nepheline syenite', 'graphic granite']

labels_name_plutonic_v3 = {0: '橄榄岩', 1: '辉石岩', 2: '角闪石岩', 3: '辉长岩', 4: '闪长岩', 5: '正长岩',
                           6: '二长岩', 7: '正长花岗岩', 8: '二长花岗岩', 9: '花岗闪长岩', 10: '斜长花岗岩',
                           11: '霞石正长岩', 12: '文象花岗岩'}


def merge_dicts(*dicts):
    result = {}
    start_i = 0
    for d in dicts:
        for k, v in d.items():
            result[k + start_i] = v
        start_i += len(d)
    return result


def get_rock_labels_name(dataset_name):
    dataset_name = dataset_name.lower()

    class_labels = []
    labels_name = {}
    if 'plutonicrocks13_v3' in dataset_name:
        class_labels = class_labels_plutonic_v3
        labels_name = labels_name_plutonic_v3

    return class_labels, labels_name


def main():
    import json

    dataset_name = 'plutonicrocks13_v3'
    class_labels, labels_name = get_rock_labels_name(dataset_name)
    print("class_labels: ", class_labels)
    # print("labels_name: ", labels_name)
    print(f"labels_name:\n{json.dumps(labels_name, indent=4, ensure_ascii=False)}")


if __name__ == '__main__':
    main()
