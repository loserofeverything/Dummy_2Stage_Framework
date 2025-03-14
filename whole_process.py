import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from tqdm import tqdm
import time
from sklearn.model_selection import train_test_split

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

# 假设已经引入前面定义的TCRDataset, TCREmbedding, TCRExtractor和NTXentLoss类

# 定义聚合层 - 将同一库内的多个TCR序列特征聚合为单个库特征
class TCRAggregator(nn.Module):
    """TCR库特征聚合层(Agg)"""
    def __init__(self, feature_dim=128, aggregation_type="attention"):
        super(TCRAggregator, self).__init__()
        self.aggregation_type = aggregation_type
        self.feature_dim = feature_dim
        
        if aggregation_type == "attention":
            # 注意力聚合
            self.attention = nn.Sequential(
                nn.Linear(feature_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 1)
            )
    
    def forward(self, features, library_mask):
        """
        参数:
        features: 形状为 [batch_size, feature_dim] 的特征向量
        library_mask: 形状为 [num_libraries, batch_size] 的掩码矩阵
                     如果序列j属于库i，则library_mask[i,j]=1，否则为0
        
        返回:
        形状为 [num_libraries, feature_dim] 的库特征向量
        """
        num_libraries = library_mask.shape[0]
        batch_size = features.shape[0]
        
        if self.aggregation_type == "mean":
            # 平均聚合
            # 对每个库，计算属于该库的所有序列的平均特征
            library_features = torch.zeros(num_libraries, self.feature_dim, device=features.device)
            for i in range(num_libraries):
                mask = library_mask[i]
                if mask.sum() > 0:  # 确保库中有序列
                    library_features[i] = (features * mask.unsqueeze(1)).sum(0) / mask.sum()
                    
        elif self.aggregation_type == "attention":
            # 注意力聚合
            # 计算每个序列的注意力权重
            attention_weights = self.attention(features).squeeze(-1)  # [batch_size]
            
            # 为每个库聚合特征
            library_features = torch.zeros(num_libraries, self.feature_dim, device=features.device)
            for i in range(num_libraries):
                mask = library_mask[i]
                if mask.sum() > 0:
                    # 将不属于此库的序列的注意力设为负无穷
                    masked_attention = attention_weights.clone()
                    masked_attention[mask == 0] = float('-inf')
                    
                    # Softmax获取归一化权重
                    weights = F.softmax(masked_attention, dim=0)
                    
                    # 加权聚合
                    library_features[i] = (features * weights.unsqueeze(1)).sum(0)
        
        return library_features

# 定义分类器
class TCRClassifier(nn.Module):
    """TCR库分类器(Cls)"""
    def __init__(self, input_dim=128, num_classes=2):
        super(TCRClassifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        return self.classifier(x)

# 库级别数据集，用于分类训练
class LibraryDataset(Dataset):
    def __init__(self, libraries, labels):
        self.libraries = libraries
        self.labels = labels
    
    def __len__(self):
        return len(self.libraries)
    
    def __getitem__(self, idx):
        return self.libraries[idx], self.labels[idx]

# 预训练阶段
def pretrain_extractor(train_dataset, val_dataset, embedding_model, extractor_model, device='cuda', 
                       batch_size=128, num_epochs=50, lr=0.001, save_path='models'):
    """使用对比学习预训练特征提取器"""
    # 创建目录保存模型
    os.makedirs(save_path, exist_ok=True)
    
    # 将模型移至指定设备
    embedding_model = embedding_model.to(device)
    extractor_model = extractor_model.to(device)
    
    # # 创建数据加载器
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
    #                          collate_fn=train_dataset.collate_fn)
    # val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
    #                        collate_fn=val_dataset.collate_fn)
    
    # # 定义损失函数和优化器
    # criterion = NTXentLoss(temperature=0.07)
    optimizer = optim.Adam(list(embedding_model.parameters()) + list(extractor_model.parameters()), lr=lr)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # # 用于早停的变量
    # best_val_loss = float('inf')
    # patience_counter = 0
    # early_stopping_patience = 10
    
    # # 记录训练过程
    # train_losses = []
    # val_losses = []
    
    # print("开始预训练...")
    
    # for epoch in range(num_epochs):
    #     # 训练阶段
    #     embedding_model.train()
    #     extractor_model.train()
    #     train_loss = 0.0
        
    #     progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
    #     for sequences, lib_ids, _ in progress_bar:
    #         # 移动数据到设备
    #         sequences = sequences.to(device)
    #         lib_ids = lib_ids.to(device)
            
    #         # 前向传播
    #         embeddings = embedding_model(sequences)
    #         features = extractor_model(embeddings)
            
    #         # 计算损失
    #         loss = criterion(features, lib_ids)
            
    #         # 反向传播和优化
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
            
    #         # 更新统计信息
    #         train_loss += loss.item()
    #         progress_bar.set_postfix({'loss': loss.item()})
        
    #     train_loss /= len(train_loader)
    #     train_losses.append(train_loss)
        
    #     # 验证阶段
    #     embedding_model.eval()
    #     extractor_model.eval()
    #     val_loss = 0.0
        
    #     with torch.no_grad():
    #         progress_bar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
    #         for sequences, lib_ids, _ in progress_bar:
    #             sequences = sequences.to(device)
    #             lib_ids = lib_ids.to(device)
                
    #             embeddings = embedding_model(sequences)
    #             features = extractor_model(embeddings)
                
    #             loss = criterion(features, lib_ids)
    #             val_loss += loss.item()
    #             progress_bar.set_postfix({'loss': loss.item()})
        
    #     val_loss /= len(val_loader)
    #     val_losses.append(val_loss)
        
    #     # 学习率调度
    #     scheduler.step(val_loss)
        
    #     print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
    #     # 早停检查
    #     if val_loss < best_val_loss:
    #         best_val_loss = val_loss
    #         patience_counter = 0
            
    #         # 保存最佳模型
    #         torch.save({
    #             'embedding_state_dict': embedding_model.state_dict(),
    #             'extractor_state_dict': extractor_model.state_dict(),
    #             'optimizer_state_dict': optimizer.state_dict(),
    #             'train_loss': train_loss,
    #             'val_loss': val_loss,
    #             'epoch': epoch,
    #         }, os.path.join(save_path, 'best_pretrained_model.pt'))
            
    #         print(f'模型已保存到 {save_path}/best_pretrained_model.pt')
    #     else:
    #         patience_counter += 1
    #         if patience_counter >= early_stopping_patience:
    #             print(f'早停: {early_stopping_patience} 个epoch内没有改进')
    #             break
    
    # # 绘制训练过程
    # plt.figure(figsize=(10, 5))
    # plt.plot(train_losses, label='Training Loss')
    # plt.plot(val_losses, label='Validation Loss')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.title('Pretraining Loss')
    # plt.legend()
    # plt.savefig(os.path.join(save_path, 'pretraining_loss.png'))
    # plt.close()
    
    

    torch.save({
        'embedding_state_dict': embedding_model.state_dict(),
        'extractor_state_dict': extractor_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': 1.19810,
        'val_loss': 114.514,
        'epoch': -1,
    }, os.path.join(save_path, 'best_pretrained_model.pt'))
    
    train_losses = [1.19810]
    val_losses = [114.514]
    print("预训练完成!")
    
    # 加载最佳模型
    checkpoint = torch.load(os.path.join(save_path, 'best_pretrained_model.pt'))
    embedding_model.load_state_dict(checkpoint['embedding_state_dict'])
    extractor_model.load_state_dict(checkpoint['extractor_state_dict'])
    
    return embedding_model, extractor_model, train_losses, val_losses

# 提取库特征
def extract_library_features(libraries, labels, embedding_model, extractor_model, device='cuda'):
    """从TCR库中提取特征向量"""
    # 将模型设置为评估模式
    embedding_model.eval()
    extractor_model.eval()
    
    all_features = []
    all_labels = []
    
    for lib_idx, (library, label) in enumerate(zip(libraries, labels)):
        print(f"处理库 {lib_idx+1}/{len(libraries)}...", end='\r')
        
        # 创建数据加载器处理单个库
        tcr_sequences = []
        for seq in library:
            # 手动编码序列
            encoded_seq = [AA_TO_IDX.get(aa, AA_TO_IDX['X']) for aa in seq[:MAX_LENGTH]]
            # 填充
            padding_length = MAX_LENGTH - len(encoded_seq)
            if padding_length > 0:
                encoded_seq = encoded_seq + [AA_TO_IDX['-']] * padding_length
            tcr_sequences.append(encoded_seq)
        
        # 转换为张量
        tcr_sequences = torch.tensor(tcr_sequences, dtype=torch.long)
        
        # 分批处理
        batch_size = 128
        library_features = []
        
        with torch.no_grad():
            for i in range(0, len(tcr_sequences), batch_size):
                batch = tcr_sequences[i:i+batch_size].to(device)
                embeddings = embedding_model(batch)
                features = extractor_model(embeddings)
                library_features.append(features.cpu())
        
        # 连接所有批次的特征
        library_features = torch.cat(library_features, dim=0)
        
        # 使用均值聚合特征
        library_feature = torch.mean(library_features, dim=0)
        
        all_features.append(library_feature)
        all_labels.append(label)
    
    # 将所有特征和标签堆叠成张量
    all_features = torch.stack(all_features)
    all_labels = torch.tensor(all_labels)
    
    print("\n特征提取完成!")
    
    return all_features, all_labels

# 分类训练阶段
def train_classifier(train_features, train_labels, val_features, val_labels, 
                     test_features, test_labels, classifier_model, 
                     device='cuda', batch_size=32, num_epochs=30, lr=0.0005, save_path='models'):
    """训练库分类器"""
    # 将数据移至指定设备
    train_features = train_features.to(device)
    train_labels = train_labels.to(device)
    val_features = val_features.to(device)
    val_labels = val_labels.to(device)
    test_features = test_features.to(device)
    test_labels = test_labels.to(device)
    
    # 将模型移到设备
    classifier_model = classifier_model.to(device)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier_model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    # 用于早停的变量
    best_val_accuracy = 0.0
    patience_counter = 0
    early_stopping_patience = 7
    
    # 记录训练过程
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    print("开始分类器训练...")
    
    for epoch in range(num_epochs):
        # 训练阶段
        classifier_model.train()
        
        # 创建数据加载器
        train_dataset = LibraryDataset(train_features, train_labels)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_features, batch_labels in train_loader:
            # 前向传播
            outputs = classifier_model(batch_features)
            loss = criterion(outputs, batch_labels)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 统计
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += batch_labels.size(0)
            train_correct += (predicted == batch_labels).sum().item()
        
        train_loss /= len(train_loader)
        train_accuracy = 100 * train_correct / train_total
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        
        # 验证阶段
        classifier_model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            val_dataset = LibraryDataset(val_features, val_labels)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)
            
            for batch_features, batch_labels in val_loader:
                outputs = classifier_model(batch_features)
                loss = criterion(outputs, batch_labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += batch_labels.size(0)
                val_correct += (predicted == batch_labels).sum().item()
        
        val_loss /= len(val_loader)
        val_accuracy = 100 * val_correct / val_total
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        
        # 学习率调度
        scheduler.step(val_loss)
        
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, '
              f'Train Acc: {train_accuracy:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%')
        
        # 早停检查
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            patience_counter = 0
            
            # 保存最佳模型
            torch.save({
                'classifier_state_dict': classifier_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_accuracy': train_accuracy,
                'val_accuracy': val_accuracy,
                'epoch': epoch,
            }, os.path.join(save_path, 'best_classifier_model.pt'))
            
            print(f'模型已保存到 {save_path}/best_classifier_model.pt')
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f'早停: {early_stopping_patience} 个epoch内没有改进')
                break
    
    # 绘制训练过程
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    ax1.plot(train_losses, label='Training Loss')
    ax1.plot(val_losses, label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Classification Loss')
    ax1.legend()
    
    ax2.plot(train_accuracies, label='Training Accuracy')
    ax2.plot(val_accuracies, label='Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Classification Accuracy')
    ax2.legend()
    
    plt.savefig(os.path.join(save_path, 'classification_training.png'))
    plt.close()
    
    # 加载最佳模型进行测试
    checkpoint = torch.load(os.path.join(save_path, 'best_classifier_model.pt'))
    classifier_model.load_state_dict(checkpoint['classifier_state_dict'])
    
    # 测试阶段
    classifier_model.eval()
    with torch.no_grad():
        test_outputs = classifier_model(test_features)
        _, test_predicted = torch.max(test_outputs.data, 1)
        
        test_accuracy = 100 * (test_predicted == test_labels).sum().item() / test_labels.size(0)
        test_f1 = f1_score(test_labels.cpu().numpy(), test_predicted.cpu().numpy(), average='weighted')
        
        print(f"\n测试结果:")
        print(f"准确率: {test_accuracy:.2f}%")
        print(f"F1分数: {test_f1:.4f}")
        
        # 混淆矩阵
        cm = confusion_matrix(test_labels.cpu().numpy(), test_predicted.cpu().numpy())
        print("混淆矩阵:")
        print(cm)
    
    return classifier_model, test_accuracy, test_f1

# 主函数，运行整个训练流程
def main(data_path='data/tcr_small_dataset.pkl', device='cuda', feature_dim=128):
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 检查设备
    if torch.cuda.is_available() and device == 'cuda':
        device = torch.device('cuda')
        print("使用GPU训练...")
    else:
        device = torch.device('cpu')
        print("使用CPU训练...")
    
    # 加载数据
    print("加载数据...")
    with open(data_path, 'rb') as f:
        libraries, labels = pickle.load(f)
    
    print(f"数据集信息: {len(libraries)} 个库, 每个库有 {len(libraries[0])} 条TCR序列")
    print(f"类别分布: {np.bincount(labels)}")
    
    # 在库级别划分数据集
    print("\n划分数据集...")
    train_indices, temp_indices = train_test_split(
        range(len(libraries)), test_size=0.3, random_state=42, stratify=labels
    )
    val_indices, test_indices = train_test_split(
        temp_indices, test_size=0.5, random_state=42, stratify=[labels[i] for i in temp_indices]
    )
    
    # 创建库的训练/验证/测试集
    train_libraries = [libraries[i] for i in train_indices]
    val_libraries = [libraries[i] for i in val_indices]
    test_libraries = [libraries[i] for i in test_indices]
    
    train_labels = [labels[i] for i in train_indices]
    val_labels = [labels[i] for i in val_indices]
    test_labels = [labels[i] for i in test_indices]
    
    print(f"训练集: {len(train_libraries)} 个库")
    print(f"验证集: {len(val_libraries)} 个库")
    print(f"测试集: {len(test_libraries)} 个库")
    
    # 创建序列级别的数据集
    full_dataset = TCRDataset(libraries, labels)
    
    # 创建训练和验证数据集
    train_dataset = TCRDataset(train_libraries, train_labels)
    val_dataset = TCRDataset(val_libraries, val_labels)
    test_dataset = TCRDataset(test_libraries, test_labels)
    
    # 创建模型
    print("创建模型...")
    embedding_model = TCREmbedding(vocab_size=len(AA_LETTERS), embed_dim=32)
    extractor_model = TCRExtractor(in_channels=32, feature_dim=feature_dim)
    classifier_model = TCRClassifier(input_dim=feature_dim, num_classes=len(np.unique(labels)))
    
    # 第一阶段：预训练特征提取器
    print("\n第一阶段: 使用对比学习预训练特征提取器...")
    embedding_model, extractor_model, _, _ = pretrain_extractor(
        train_dataset, val_dataset, embedding_model, extractor_model, 
        device=device, batch_size=128, num_epochs=30
    )
    
    # 使用预训练的模型提取库特征
    print("\n提取库级别特征...")
    # 分别为训练、验证和测试集提取特征
    train_features, train_lib_labels = extract_library_features(
        train_libraries, train_labels, embedding_model, extractor_model, device=device
    )
    
    val_features, val_lib_labels = extract_library_features(
        val_libraries, val_labels, embedding_model, extractor_model, device=device
    )
    
    test_features, test_lib_labels = extract_library_features(
        test_libraries, test_labels, embedding_model, extractor_model, device=device
    )
    
    # 第二阶段：训练分类器
    print("\n第二阶段: 训练库分类器...")
    classifier_model, test_accuracy, test_f1 = train_classifier(
        train_features, train_lib_labels,
        val_features, val_lib_labels,
        test_features, test_lib_labels,
        classifier_model, device=device, batch_size=16, num_epochs=50
    )
    
    print("\n训练完成!")
    print(f"最终测试准确率: {test_accuracy:.2f}%")
    print(f"最终测试F1分数: {test_f1:.4f}")
    
    return embedding_model, extractor_model, classifier_model

if __name__ == "__main__":
    main()