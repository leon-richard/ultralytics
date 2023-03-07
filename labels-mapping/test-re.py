import re

# 定义要处理的文本
text = """-1 hello
2 world
-1 123"""

# 定义正则表达式，匹配每行开头的整数
pattern = r"(^\d+)|(^-1\s+)"

# 定义一个函数，用于将每行匹配到的整数加一
def increment(match):
    if match.group(1):
        num = int(match.group(1))
        return str(num + 1)
    else:
        return ""

# 使用sub()函数替换每行的整数为自身加一，并删除行首整数为-1的行
result = re.sub(pattern, increment, text, flags=re.MULTILINE)

# 删除被删除行的换行符
result = re.sub(r"\n\n", "\n", result)

# 输出处理后的结果
print(result)
