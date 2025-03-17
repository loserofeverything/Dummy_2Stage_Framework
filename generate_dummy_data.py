import numpy as np
import random
import pickle
import os

# 氨基酸字母表（20种常见氨基酸）
AA_LETTERS = "ACDEFGHIKLMNPQRSTVWY"

def generate_tcr_sequence_without_patterns(min_length=8, max_length=16, forbidden_patterns=None):
    """生成一条随机的TCR CDR3序列，确保不包含任何指定的模式"""
    if not forbidden_patterns:
        forbidden_patterns = []
    
    max_attempts = 100  # 防止无限循环
    for _ in range(max_attempts):
        length = random.randint(min_length, max_length)
        seq = ''.join(random.choice(AA_LETTERS) for _ in range(length))
        
        # 检查序列是否包含任何禁止的模式
        if not any(pattern in seq for pattern in forbidden_patterns):
            return seq
    
    # 如果多次尝试后仍无法生成，则强制替换可能的模式
    length = random.randint(min_length, max_length)
    seq = ''.join(random.choice(AA_LETTERS) for _ in range(length))
    for pattern in forbidden_patterns:
        while pattern in seq:
            replacement = ''.join(random.choice(AA_LETTERS) for _ in range(len(pattern)))
            seq = seq.replace(pattern, replacement, 1)
    return seq

def generate_tcr_sequence(min_length=8, max_length=16):
    """生成一条随机的TCR CDR3序列"""
    length = random.randint(min_length, max_length)
    return ''.join(random.choice(AA_LETTERS) for _ in range(length))

def generate_biased_tcr(pattern, min_length=8, max_length=16, forbidden_patterns=None):
    """生成包含特定模式的TCR序列，同时确保不包含其他禁止的模式"""
    if forbidden_patterns is None:
        forbidden_patterns = []
    
    # 移除当前模式(如果存在于禁止列表中)
    safe_forbidden = [p for p in forbidden_patterns if p != pattern]
    
    max_attempts = 100
    for _ in range(max_attempts):
        # 确保基础序列足够长
        length = random.randint(max(min_length, len(pattern) + 2), max_length)
        base_seq = ''.join(random.choice(AA_LETTERS) for _ in range(length))
        
        # 检查基础序列是否包含任何禁止的模式
        if any(p in base_seq for p in safe_forbidden):
            continue
        
        # 在随机位置插入目标模式
        pos = random.randint(0, len(base_seq) - len(pattern))
        seq = base_seq[:pos] + pattern + base_seq[pos+len(pattern):]
        if len(seq) > max_length:
            seq = seq[:max_length]
        
        # 最后检查一次是否包含禁止模式
        if not any(p in seq for p in safe_forbidden):
            return seq
    
    # 如果达到最大尝试次数，使用更简单的方法
    min_viable_length = len(pattern) + 2
    length = min(max_length, min_viable_length + 4)
    remaining = length - len(pattern)
    prefix_len = random.randint(1, remaining - 1)
    suffix_len = remaining - prefix_len
    
    prefix = ''.join(random.choice(AA_LETTERS) for _ in range(prefix_len))
    suffix = ''.join(random.choice(AA_LETTERS) for _ in range(suffix_len))
    return prefix + pattern + suffix

def generate_tcr_dataset(n_libraries=2100, tcrs_per_library=10000, n_classes=2, pattern_ratio=0.3, seed=42):
    """
    生成一个TCR库数据集
    
    参数:
    - n_libraries: 库的数量
    - tcrs_per_library: 每个库中TCR序列的数量
    - n_classes: 类别数量
    - pattern_ratio: 包含模式的序列比例
    - seed: 随机种子，确保可重复性
    
    返回:
    - libraries: 列表，每个元素是一个库(由多个TCR序列组成)
    - labels: 列表，每个库的标签
    - class_patterns: 列表，每个类别的特征模式
    """
    random.seed(seed)
    np.random.seed(seed)
    
    libraries = []
    labels = []
    
    # 为不同类别创建特征模式，以模拟生物学相关性
    class_patterns = []
    all_patterns = []
    for c in range(n_classes):
        # 每个类别有3-5个特征模式
        n_patterns = random.randint(3, 5)
        patterns = []
        for _ in range(n_patterns):
            # 每个模式长度为4-7个氨基酸
            pattern_len = random.randint(4, 7)
            pattern = ''.join(random.choice(AA_LETTERS) for _ in range(pattern_len))
            patterns.append(pattern)
            all_patterns.append(pattern)
        class_patterns.append(patterns)
    
    for i in range(n_libraries):
        class_label = i % n_classes
        library = []
        
        # 该类别的特征模式
        current_patterns = class_patterns[class_label]
        
        # 其他类别的模式(需要排除)
        other_patterns = [p for c_idx, patterns in enumerate(class_patterns) 
                          for p in patterns if c_idx != class_label]
        
        # 生成该库的所有TCR序列
        pattern_count = int(tcrs_per_library * pattern_ratio)
        
        # 生成包含模式的TCR序列
        for j in range(pattern_count):
            pattern = random.choice(current_patterns)
            tcr = generate_biased_tcr(pattern, forbidden_patterns=other_patterns)
            library.append(tcr)
            
        # 生成不包含任何模式的TCR序列
        for j in range(tcrs_per_library - pattern_count):
            tcr = generate_tcr_sequence_without_patterns(forbidden_patterns=all_patterns)
            library.append(tcr)
        
        # 打乱序列顺序
        random.shuffle(library)    
        libraries.append(library)
        labels.append(class_label)
    
    return libraries, labels, class_patterns

# 生成小型样例数据集用于演示
def create_sample_dataset(pattern_ratio=0.3):
    # 创建一个较小的数据集用于测试
    small_libraries, small_labels, small_patterns = generate_tcr_dataset(
        n_libraries=200,  # 200个样本库
        tcrs_per_library=5000,  # 每个库5000条序列
        n_classes=4,  # 四分类问题
        pattern_ratio=pattern_ratio  # 包含模式的序列比例
    )
    
    # 打印数据集信息
    print(f"生成了 {len(small_libraries)} 个库，每个库包含 {len(small_libraries[0])} 条TCR序列")
    print(f"类别分布: {np.bincount(small_labels)}")
    print(f"每个库中约有 {pattern_ratio*100:.1f}% 的序列包含该类特征模式")
    
    # 显示部分样本
    print("\n前3个库的前3条序列样例:")
    for i in range(min(3, len(small_libraries))):
        print(f"库 {i} (标签: {small_labels[i]}): {small_libraries[i][:3]}")
    
    # 显示类别模式
    print("\n各类别的特征模式:")
    for i, patterns in enumerate(small_patterns):
        print(f"类别 {i}: {patterns}")
    
    # 保存数据集
    os.makedirs("data", exist_ok=True)
    with open("data/tcr_small_dataset.pkl", "wb") as f:
        pickle.dump((small_libraries, small_labels), f)
    print("\n样本数据集已保存到 data/tcr_small_dataset.pkl")
    
    # 保存类别模式
    with open("data/tcr_small_patterns.pkl", "wb") as f:
        pickle.dump(small_patterns, f)
    print("类别特征模式已保存到 data/tcr_small_patterns.pkl")
    
    return small_libraries, small_labels, small_patterns

if __name__ == "__main__":
    create_sample_dataset(pattern_ratio=0.3)
    
    # 如需生成完整数据集，取消下面的注释
    # print("生成完整数据集...")
    # libraries, labels, patterns = generate_tcr_dataset(pattern_ratio=0.3)
    # with open("data/tcr_full_dataset.pkl", "wb") as f:
    #     pickle.dump((libraries, labels), f)
    # print("完整数据集已保存到 data/tcr_full_dataset.pkl")