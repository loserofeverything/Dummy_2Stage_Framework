import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
from torch.utils.data import Dataset, DataLoader

# 定义常量
AA_LETTERS = "-ACDEFGHIKLMNPQRSTVWYX"  # 添加填充符"-"和未知氨基酸"X"
MAX_LENGTH = 20  # TCR序列最大长度

# 创建氨基酸到索引的映射
AA_TO_IDX = {aa: idx for idx, aa in enumerate(AA_LETTERS)}

class TCRDataset(Dataset):
    """TCR序列数据集类"""
    def __init__(self, libraries, labels=None, max_length=MAX_LENGTH):
        self.libraries = libraries
        self.labels = labels
        self.max_length = max_length
        
        # 展平库，记录每个序列属于哪个库
        self.sequences = []
        self.library_ids = []
        for lib_id, library in enumerate(libraries):
            self.sequences.extend(library)
            self.library_ids.extend([lib_id] * len(library))
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        lib_id = self.library_ids[idx]
        
        # 序列编码
        encoded_seq = self.encode_sequence(seq)
        
        if self.labels is not None:
            label = self.labels[lib_id]
            return encoded_seq, lib_id, label
        else:
            return encoded_seq, lib_id
    
    def encode_sequence(self, seq):
        """将氨基酸序列编码为整数索引序列，并进行填充"""
        # 将序列转换为整数索引
        encoded = [AA_TO_IDX.get(aa, AA_TO_IDX['X']) for aa in seq[:self.max_length]]
        
        # 填充序列
        padding_length = self.max_length - len(encoded)
        if padding_length > 0:
            encoded = encoded + [AA_TO_IDX['-']] * padding_length
            
        return torch.tensor(encoded, dtype=torch.long)

    @staticmethod
    def collate_fn(batch):
        """自定义批处理函数"""
        sequences, lib_ids, labels = zip(*batch)
        sequences = torch.stack(sequences)
        lib_ids = torch.tensor(lib_ids)
        labels = torch.tensor(labels)
        return sequences, lib_ids, labels

class TCREmbedding(nn.Module):
    """TCR序列的嵌入层(Emb)"""
    def __init__(self, vocab_size=len(AA_LETTERS), embed_dim=32):
        super(TCREmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
    
    def forward(self, x):
        """
        参数:
        x: 形状为 [batch_size, seq_len] 的整数张量
        
        返回:
        形状为 [batch_size, seq_len, embed_dim] 的嵌入向量
        """
        return self.embedding(x)

class TCRExtractor(nn.Module):
    """TCR序列的特征提取网络(Ext)"""
    def __init__(self, in_channels=32, feature_dim=128):
        super(TCRExtractor, self).__init__()
        
        # 多尺度卷积层，捕获不同长度的模式
        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels, 64, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(in_channels, 64, kernel_size=7, padding=3)
        
        # 池化后的特征融合层
        self.fusion = nn.Sequential(
            nn.Conv1d(192, 128, kernel_size=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        
        # 最终特征映射层
        self.fc = nn.Sequential(
            nn.Linear(128, feature_dim),
            nn.BatchNorm1d(feature_dim)
        )
        
    def forward(self, x):
        """
        参数:
        x: 形状为 [batch_size, seq_len, embed_dim] 的嵌入向量
        
        返回:
        形状为 [batch_size, feature_dim] 的特征向量
        """
        # 调整维度顺序以适应卷积 [batch, embed_dim, seq_len]
        x = x.transpose(1, 2)
        
        # 多尺度卷积
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x))
        x3 = F.relu(self.conv3(x))
        
        # 连接多尺度特征
        x = torch.cat([x1, x2, x3], dim=1)
        
        # 特征融合
        x = self.fusion(x)
        
        # 全局最大池化
        x = F.adaptive_max_pool1d(x, 1).squeeze(2)
        
        # 最终特征映射
        x = self.fc(x)
        
        return x

# 对比学习损失函数
class NTXentLoss(nn.Module):
    """归一化温度尺度交叉熵损失，用于对比学习"""
    def __init__(self, temperature=0.5):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        
    def forward(self, features, lib_ids):
        """
        参数:
        features: 形状为 [batch_size, feature_dim] 的特征向量
        lib_ids: 形状为 [batch_size] 的库ID
        
        返回:
        对比学习损失
        """
        # 归一化特征向量
        features = F.normalize(features, dim=1)
        
        # 计算所有样本对之间的相似度
        sim_matrix = torch.matmul(features, features.T) / self.temperature
        
        # 创建标签矩阵：如果两个样本来自同一库，则为正样本对
        labels = (lib_ids.unsqueeze(1) == lib_ids.unsqueeze(0)).float()
        
        # 移除自身相似度
        mask = torch.eye(labels.shape[0], dtype=torch.bool, device=labels.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        sim_matrix = sim_matrix[~mask].view(sim_matrix.shape[0], -1)
        
        # 正样本对的相似度应该比负样本对高
        loss = self.criterion(sim_matrix, labels)
        
        return loss / labels.shape[0]

# 简单测试模型
def test_model(libraries, labels):
    # 创建数据集和数据加载器
    dataset = TCRDataset(libraries, labels)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # 创建模型
    embedding = TCREmbedding()
    extractor = TCRExtractor()
    loss_fn = NTXentLoss()
    
    # 测试一个批次
    sequences, lib_ids, _ = next(iter(dataloader))
    embeddings = embedding(sequences)
    features = extractor(embeddings)
    
    # 计算损失
    loss = loss_fn(features, lib_ids)
    
    print(f"输入序列形状: {sequences.shape}")
    print(f"嵌入向量形状: {embeddings.shape}")
    print(f"特征向量形状: {features.shape}")
    print(f"对比学习损失: {loss.item()}")
    
    return embedding, extractor

# 加载示例数据并测试模型
if __name__ == "__main__":
    with open("data/tcr_small_dataset.pkl", "rb") as f:
        libraries, labels = pickle.load(f)
    
    embedding, extractor = test_model(libraries, labels)
    print("模型测试完成!")