import numpy as np
import random
import pickle
import os

# 氨基酸字母表（20种常见氨基酸）
AA_LETTERS = "ACDEFGHIKLMNPQRSTVWY"

def generate_tcr_sequence(min_length=8, max_length=16):
    """生成一条随机的TCR CDR3序列"""
    length = random.randint(min_length, max_length)
    return ''.join(random.choice(AA_LETTERS) for _ in range(length))

def generate_biased_tcr(pattern, min_length=8, max_length=16):
    """生成包含特定模式的TCR序列，用于创建有生物学意义的数据"""
    length = random.randint(min_length, max_length)
    seq = ''.join(random.choice(AA_LETTERS) for _ in range(length))
    
    # 在随机位置插入模式
    if len(seq) > len(pattern):
        pos = random.randint(0, len(seq) - len(pattern))
        seq = seq[:pos] + pattern + seq[pos+len(pattern):]
        if len(seq) > max_length:
            seq = seq[:max_length]
    return seq

def generate_tcr_dataset(n_libraries=2100, tcrs_per_library=10000, n_classes=2, seed=42):
    """
    生成一个TCR库数据集
    
    参数:
    - n_libraries: 库的数量
    - tcrs_per_library: 每个库中TCR序列的数量
    - n_classes: 类别数量
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
    for c in range(n_classes):
        # 每个类别有3-5个特征模式
        n_patterns = random.randint(3, 5)
        patterns = []
        for _ in range(n_patterns):
            # 每个模式长度为3-6个氨基酸
            pattern_len = random.randint(4, 7)
            pattern = ''.join(random.choice(AA_LETTERS) for _ in range(pattern_len))
            patterns.append(pattern)
        class_patterns.append(patterns)
    
    for i in range(n_libraries):
        class_label = i % n_classes
        library = []
        
        # 该类别的特征模式
        patterns = class_patterns[class_label]
        
        # 生成该库的所有TCR序列
        for j in range(tcrs_per_library):
            # 70%的TCR包含该类的特征模式之一
            if random.random() < 0.01:
                pattern = random.choice(patterns)
                tcr = generate_biased_tcr(pattern)
            else:
                tcr = generate_tcr_sequence()
            library.append(tcr)
            
        libraries.append(library)
        labels.append(class_label)
    
    return libraries, labels, class_patterns

# 生成小型样例数据集用于演示
def create_sample_dataset():
    # 创建一个较小的数据集用于测试
    small_libraries, small_labels, small_patterns = generate_tcr_dataset(
        n_libraries=200,  # 20个样本库
        tcrs_per_library=5000,  # 每个库100条序列
        n_classes=4  # 二分类问题
    )
    
    # 打印数据集信息
    print(f"生成了 {len(small_libraries)} 个库，每个库包含 {len(small_libraries[0])} 条TCR序列")
    print(f"类别分布: {np.bincount(small_labels)}")
    
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
    create_sample_dataset()
    
    # 如需生成完整数据集，取消下面的注释
    # print("生成完整数据集...")
    # libraries, labels = generate_tcr_dataset()
    # with open("data/tcr_full_dataset.pkl", "wb") as f:
    #     pickle.dump((libraries, labels), f)
    # print("完整数据集已保存到 data/tcr_full_dataset.pkl")