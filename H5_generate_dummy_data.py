import numpy as np
import random
import os
import h5py
import uuid
import json
import datetime

# 氨基酸字母表（20种常见氨基酸）
AA_LETTERS = "ACDEFGHIKLMNPQRSTVWY"

def generate_sample_id():
    """生成一个随机样本ID"""
    return f"sample_{uuid.uuid4().hex[:8]}"

def generate_timestamp_dir():
    """生成基于当前时间的目录名"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"dataset_{timestamp}"

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

def generate_tcr_dataset(n_samples=2100, tcrs_per_sample=10000, n_classes=2, pattern_ratio=0.3, seed=42, class_patterns=None):
    """
    生成一个TCR库数据集
    
    参数:
    - n_samples: 样本的数量
    - tcrs_per_sample: 每个样本中TCR序列的数量
    - n_classes: 类别数量
    - pattern_ratio: 包含模式的序列比例
    - seed: 随机种子，确保可重复性
    - class_patterns: 如果提供，使用这些类别模式而非生成新的
    
    返回:
    - sample_ids: 样本ID列表
    - labels: 列表，每个样本的标签
    - class_patterns: 列表，每个类别的特征模式
    """
    random.seed(seed)
    np.random.seed(seed)
    
    sample_ids = []
    labels = []
    tcr_libraries = []
    
    # 如果没有提供类别模式，则生成新的
    if class_patterns is None:
        class_patterns = []
        all_patterns = []
        for c in range(n_classes):
            # 第4类(索引为3)不生成特定模式，作为控制组
            if c == 3 and n_classes > 3:  # 确保有第4类
                class_patterns.append([])  # 空模式列表
                continue
                
            # 其他类别有3-5个特征模式
            n_patterns = random.randint(3, 5)
            patterns = []
            for _ in range(n_patterns):
                # 每个模式长度为4-7个氨基酸
                pattern_len = random.randint(4, 7)
                pattern = ''.join(random.choice(AA_LETTERS) for _ in range(pattern_len))
                patterns.append(pattern)
                all_patterns.append(pattern)
            class_patterns.append(patterns)
    else:
        # 使用提供的类别模式
        all_patterns = [pattern for patterns in class_patterns for pattern in patterns]
    
    for i in range(n_samples):
        # 生成唯一样本ID
        sample_id = generate_sample_id()
        sample_ids.append(sample_id)
        
        class_label = i % n_classes
        labels.append(class_label)
        library = []
        
        # 该类别的特征模式
        current_patterns = class_patterns[class_label]
        
        # 其他类别的模式(需要排除)
        other_patterns = [p for c_idx, patterns in enumerate(class_patterns) 
                          for p in patterns if c_idx != class_label]
        
        # 第4类(控制组)全部生成无模式序列
        if class_label == 3 and n_classes > 3:
            for j in range(tcrs_per_sample):
                tcr = generate_tcr_sequence_without_patterns(forbidden_patterns=all_patterns)
                library.append(tcr)
        else:
            # 非控制组生成含模式和不含模式的序列
            pattern_count = int(tcrs_per_sample * pattern_ratio)
            
            # 生成包含模式的TCR序列
            for j in range(pattern_count):
                if current_patterns:  # 确保有可用模式
                    pattern = random.choice(current_patterns)
                    tcr = generate_biased_tcr(pattern, forbidden_patterns=other_patterns)
                    library.append(tcr)
                else:
                    # 没有模式的类别，生成不含模式的序列
                    tcr = generate_tcr_sequence_without_patterns(forbidden_patterns=all_patterns)
                    library.append(tcr)
                    
            # 生成不包含任何模式的TCR序列
            for j in range(tcrs_per_sample - pattern_count):
                tcr = generate_tcr_sequence_without_patterns(forbidden_patterns=all_patterns)
                library.append(tcr)
        
        # 打乱序列顺序
        random.shuffle(library)
        tcr_libraries.append(library)
    
    return sample_ids, labels, tcr_libraries, class_patterns

def save_to_h5(file_path, sample_ids, labels, tcr_libraries):
    """将数据保存为HDF5格式"""
    with h5py.File(file_path, 'w') as h5file:
        for i, sample_id in enumerate(sample_ids):
            # 为每个样本创建一个组
            sample_group = h5file.create_group(sample_id)
            
            # 添加kmer_labels数据集 - 形状为(k,)，值都为样本标签
            label = labels[i]
            kmer_count = len(tcr_libraries[i])
            kmer_labels = np.full(kmer_count, label, dtype=np.int32)
            sample_group.create_dataset('kmer_labels', data=kmer_labels)
            
            # 添加kmers数据集 - 保存所有TCR序列
            kmers = np.array(tcr_libraries[i], dtype='S20')  # 使用S20表示最多20个字符的字符串
            sample_group.create_dataset('kmers', data=kmers)
            
            # 额外保存一个标量label作为属性
            sample_group.attrs['label'] = label

def save_config_info(config_path, dataset_params, class_patterns, split_info=None):
    """保存数据集的配置信息"""
    # 将类模式转换为可序列化的格式
    serializable_patterns = []
    for class_idx, patterns in enumerate(class_patterns):
        serializable_patterns.append({
            "class": class_idx,
            "patterns": patterns,
            "is_control": class_idx == 3 and len(patterns) == 0
        })
    
    # 创建配置字典
    config = {
        "creation_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_parameters": dataset_params,
        "class_patterns": serializable_patterns,
        "control_class": {
            "class_id": 3,
            "description": "控制类，不包含任何特定模式"
        }
    }
    
    # 如果有数据集分割信息，也添加到配置中
    if split_info:
        config["split_info"] = split_info
    
    # 保存到JSON文件
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)

def create_split_datasets(pattern_ratio=0.3, n_samples=2100, tcrs_per_sample=10000, n_classes=4, 
                          split_ratio=(0.8, 0.1, 0.1), seed=42):
    """
    生成训练集、验证集和测试集
    
    参数:
    - pattern_ratio: 包含模式的序列比例
    - n_samples: 总样本数量
    - tcrs_per_sample: 每个样本中TCR序列的数量
    - n_classes: 类别数量
    - split_ratio: 元组(train, val, test)，表示数据集分割比例
    - seed: 随机种子
    
    返回:
    - dataset_dir: 数据集保存目录
    """
    # 创建时间戳目录
    timestamp_dir = generate_timestamp_dir()
    base_dir = "h5_data"
    dataset_dir = os.path.join(base_dir, timestamp_dir)
    os.makedirs(dataset_dir, exist_ok=True)
    
    # 验证分割比例是否合法
    train_ratio, val_ratio, test_ratio = split_ratio
    if not np.isclose(sum(split_ratio), 1.0):
        raise ValueError("分割比例之和必须等于1.0")
    
    # 确保有至少4个类别以支持控制类
    if n_classes < 4:
        n_classes = 4
        print(f"警告: 类别数量已调整为 {n_classes} 以支持控制类(标签3)")
    
    # 计算每个数据集的样本数量
    train_samples = int(n_samples * train_ratio)
    val_samples = int(n_samples * val_ratio)
    test_samples = n_samples - train_samples - val_samples
    
    # 记录数据集参数
    dataset_params = {
        "total_samples": n_samples,
        "tcrs_per_sample": tcrs_per_sample,
        "n_classes": n_classes,
        "pattern_ratio": pattern_ratio,
        "control_class": 3,  # 第4类(索引3)作为控制类
        "seed": seed
    }
    
    split_info = {
        "train_samples": train_samples,
        "val_samples": val_samples,
        "test_samples": test_samples,
        "split_ratio": list(split_ratio)  # 转为列表以便JSON序列化
    }
    
    # 设置随机种子
    random.seed(seed)
    np.random.seed(seed)
    
    # 为所有数据集生成共享的类别模式
    class_patterns = []
    all_patterns = []
    for c in range(n_classes):
        # 第4类(索引为3)不生成特定模式，作为控制组
        if c == 3:  # 第4类
            class_patterns.append([])  # 空模式列表
            print(f"类别 {c} 被设置为控制类，不含任何特征模式")
            continue
            
        # 其他类别有3-5个特征模式
        n_patterns = random.randint(3, 5)
        patterns = []
        for _ in range(n_patterns):
            # 每个模式长度为4-7个氨基酸
            pattern_len = random.randint(4, 7)
            pattern = ''.join(random.choice(AA_LETTERS) for _ in range(pattern_len))
            patterns.append(pattern)
            all_patterns.append(pattern)
        class_patterns.append(patterns)
    
    # 生成并保存训练集
    print(f"正在生成训练集 ({train_samples} 个样本)...")
    train_ids, train_labels, train_libraries, _ = generate_tcr_dataset(
        n_samples=train_samples,
        tcrs_per_sample=tcrs_per_sample,
        n_classes=n_classes,
        pattern_ratio=pattern_ratio,
        seed=seed,
        class_patterns=class_patterns  # 使用共享的类别模式
    )
    train_file_path = os.path.join(dataset_dir, "train.h5")
    save_to_h5(train_file_path, train_ids, train_labels, train_libraries)
    
    # 生成并保存验证集
    print(f"正在生成验证集 ({val_samples} 个样本)...")
    val_ids, val_labels, val_libraries, _ = generate_tcr_dataset(
        n_samples=val_samples,
        tcrs_per_sample=tcrs_per_sample,
        n_classes=n_classes,
        pattern_ratio=pattern_ratio,
        seed=seed + 1,  # 使用不同的种子避免与训练集重复
        class_patterns=class_patterns  # 使用共享的类别模式
    )
    val_file_path = os.path.join(dataset_dir, "val.h5")
    save_to_h5(val_file_path, val_ids, val_labels, val_libraries)
    
    # 生成并保存测试集
    print(f"正在生成测试集 ({test_samples} 个样本)...")
    test_ids, test_labels, test_libraries, _ = generate_tcr_dataset(
        n_samples=test_samples,
        tcrs_per_sample=tcrs_per_sample,
        n_classes=n_classes,
        pattern_ratio=pattern_ratio,
        seed=seed + 2,  # 使用不同的种子避免与训练集和验证集重复
        class_patterns=class_patterns  # 使用共享的类别模式
    )
    test_file_path = os.path.join(dataset_dir, "test.h5")
    save_to_h5(test_file_path, test_ids, test_labels, test_libraries)
    
    # 打印数据集统计信息
    print("\n数据集统计:")
    print(f"- 训练集: {len(train_ids)} 个样本, 类别分布: {np.bincount(train_labels)}")
    print(f"- 验证集: {len(val_ids)} 个样本, 类别分布: {np.bincount(val_labels)}")
    print(f"- 测试集: {len(test_ids)} 个样本, 类别分布: {np.bincount(test_labels)}")
    
    # 显示类别模式
    print("\n各类别的特征模式:")
    for i, patterns in enumerate(class_patterns):
        if i == 3:
            print(f"类别 {i}: 控制类 (不含任何特征模式)")
        else:
            print(f"类别 {i}: {patterns}")
    
    # 保存配置信息
    config_file_path = os.path.join(dataset_dir, "dataset_config.json")
    save_config_info(config_file_path, dataset_params, class_patterns, split_info)
    
    print(f"\n数据集已保存到目录: {dataset_dir}")
    print(f"- 训练集: train.h5")
    print(f"- 验证集: val.h5")
    print(f"- 测试集: test.h5")
    print(f"- 配置文件: dataset_config.json")
    
    return dataset_dir

# 保留原有的单数据集生成函数，但也支持控制类
def create_sample_dataset(pattern_ratio=0.3, n_samples=200, tcrs_per_sample=1000, n_classes=4):
    # 确保有足够的类别支持控制类
    if n_classes < 4:
        n_classes = 4
        print(f"警告: 类别数量已调整为 {n_classes} 以支持控制类(标签3)")
        
    # 创建时间戳目录
    timestamp_dir = generate_timestamp_dir()
    base_dir = "h5_data"
    dataset_dir = os.path.join(base_dir, timestamp_dir)
    os.makedirs(dataset_dir, exist_ok=True)
    
    # 记录数据集参数
    dataset_params = {
        "n_samples": n_samples,
        "tcrs_per_sample": tcrs_per_sample,
        "n_classes": n_classes,
        "pattern_ratio": pattern_ratio,
        "control_class": 3  # 第4类作为控制类
    }
    
    # 创建数据集
    sample_ids, labels, tcr_libraries, class_patterns = generate_tcr_dataset(
        n_samples=n_samples,
        tcrs_per_sample=tcrs_per_sample,
        n_classes=n_classes,
        pattern_ratio=pattern_ratio
    )
    
    # 打印数据集信息
    print(f"生成了 {len(sample_ids)} 个样本，每个样本包含 {len(tcr_libraries[0])} 条TCR序列")
    print(f"类别分布: {np.bincount(labels)}")
    print(f"每个样本中约有 {pattern_ratio*100:.1f}% 的序列包含该类特征模式 (除控制类外)")
    
    # 显示部分样本
    print("\n前3个样本的示例:")
    for i in range(min(3, len(sample_ids))):
        print(f"样本ID: {sample_ids[i]} (标签: {labels[i]})")
        print(f"  TCR序列示例: {tcr_libraries[i][:3]}")
    
    # 显示类别模式
    print("\n各类别的特征模式:")
    for i, patterns in enumerate(class_patterns):
        if i == 3:
            print(f"类别 {i}: 控制类 (不含任何特征模式)")
        else:
            print(f"类别 {i}: {patterns}")
    
    # 保存数据集为HDF5格式
    h5_file_path = os.path.join(dataset_dir, "tcr_dataset.h5")
    save_to_h5(h5_file_path, sample_ids, labels, tcr_libraries)
    
    # 保存配置信息
    config_file_path = os.path.join(dataset_dir, "dataset_config.json")
    save_config_info(config_file_path, dataset_params, class_patterns)
    
    print(f"\n数据集已保存到目录: {dataset_dir}")
    print(f"- 数据文件: {os.path.basename(h5_file_path)}")
    print(f"- 配置文件: {os.path.basename(config_file_path)}")
    
    # 返回关键信息
    return sample_ids, labels, class_patterns, dataset_dir

if __name__ == "__main__":
    # 创建训练/验证/测试数据集
    create_split_datasets(
        pattern_ratio=0.1,
        n_samples=1000,  # 总共1000个样本
        tcrs_per_sample=1000,  # 每个样本1000条TCR序列
        n_classes=4,  # 四分类问题
        split_ratio=(0.7, 0.15, 0.15)  # 70%训练，15%验证，15%测试
    )
    
    # 如果只想生成单个数据集，可以使用以下代码
    # create_sample_dataset(pattern_ratio=0.3)
    
    # 如需生成完整数据集，取消下面的注释
    # print("\n生成完整数据集...")
    # create_split_datasets(
    #     pattern_ratio=0.3,
    #     n_samples=2100,
    #     tcrs_per_sample=10000,
    #     n_classes=4,
    #     split_ratio=(0.8, 0.1, 0.1)
    # )