import re
from pathlib import Path
import yaml

from ultralytics.yolo.utils import ROOT, yaml_load

# 替换规则是：
# 1. 如果from-name与to-name匹配，则把class_id替换成新的；
# 2. 如果在to-names中找不到匹配项，则删除该标签行。

# Constants
LABELS_MAPPING_PATH = ROOT.parent / 'labels-mapping'
MAPPING_CFG_PATH = LABELS_MAPPING_PATH / 'mapping-cfg.yaml'
MAPPING_CFG_DICT = yaml_load(MAPPING_CFG_PATH)

FROM_LABELS_PATH = LABELS_MAPPING_PATH / MAPPING_CFG_DICT['from-labels-path']
TO_LABELS_PATH = LABELS_MAPPING_PATH / MAPPING_CFG_DICT['to-labels-path']

# mkdir
TO_LABELS_PATH.mkdir(parents=True, exist_ok=True)

# 解析from-names和to-names，形成mapping表
labels_mapping = {}
from_names_dict = MAPPING_CFG_DICT['from-names']
to_names_dict = MAPPING_CFG_DICT['to-names']
for from_key, from_val in from_names_dict.items():
    to_key = -1
    for k, to_val in to_names_dict.items():
        if from_val == to_val:
            to_key = k
            break
    labels_mapping[from_key] = to_key

print(labels_mapping)

# fileList = list(FROM_LABELS_PATH.rglob("*.txt"))
# fileList.sort()

# for item in fileList:
#     if item.is_file():
#         print(item)

# 统计目录中有多少个txt文件的示例
def count_txt_files(path):
    count = 0
    for file_path in path.rglob('*.txt'):
        if file_path.is_file():
            count += 1
    return count

from_label_files_num = count_txt_files(FROM_LABELS_PATH)

# 定义一个函数，用于将每行匹配到的整数加一
def map(match):
    from_class_id = int(match.group(0))
    to_class_id = labels_mapping[from_class_id]
    return str(to_class_id)

# 定义正则表达式，匹配每行开头的整数
pattern = r"^\d+"

# 逐个文件替换class_id
file_id = 0
for from_label_file in FROM_LABELS_PATH.rglob("*.txt"):
    if from_label_file.is_file():
        file_id+=1
        print(from_label_file.name, f"\t\t\t{file_id}//{from_label_files_num}")
        with open(from_label_file, mode='r', errors='ignore', encoding='utf-8') as fread:
            s = fread.read()  # string
            from_record_num = len(s.splitlines())
            # 使用sub()函数替换每行的整数为自身加一
            result = re.sub(pattern, map, s, flags=re.MULTILINE)
            # 使用sub()函数将行首整数为-1的行删除（包括换行符\n）
            result = re.sub(r"^(-1.*)$\n", "", result, flags=re.MULTILINE)

            to_record_num = len(result.splitlines())
            print(f"{from_record_num} --> {to_record_num} \tDelete {from_record_num - to_record_num} records")

            # 保存到TO_LABELS_PATH目录下，文件名不变
            to_label_file = TO_LABELS_PATH / from_label_file.name
            with open(to_label_file, mode='w') as fwrite:
                fwrite.write(result)
