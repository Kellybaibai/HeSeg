import torch
import torch.nn as nn
import torch.optim as optim
from dgl.data import CoraGraphDataset
from dgl import remove_self_loop, add_self_loop
# from sklearn.metrics import roc_auc_score
import numpy as np
import scipy.sparse as sp

from utils import *

# 检查CUDA是否可用，然后设置device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

np.random.seed(20)

def calculate_auc(labels, scores):
    sorted_indices = np.argsort(scores)
    sorted_labels = labels[sorted_indices]
    pos = sum(labels)
    neg = len(labels) - pos
    cum_neg = np.cumsum(1 - sorted_labels)
    auc = sum(cum_neg * sorted_labels) / (pos * neg)
    return auc

def sample_k_neighbors(adj, k):
    """
    对于给定的邻接矩阵adj，每个节点随机保留k个邻居。
    
    Args:
    adj (torch.Tensor): 输入的邻接矩阵，假设是稀疏表示。
    k (int): 每个节点要保留的邻居数量。
    
    Returns:
    torch.Tensor: 修改后的邻接矩阵。
    """
    # 初始化一个空的邻接矩阵
    new_adj = torch.zeros_like(adj)
    
    for i in range(adj.size(0)):
        neighbors = adj[i].nonzero(as_tuple=True)[0]  # 找到非零元素的索引
        if neighbors.size(0) > k:
            # 如果邻居数量大于k，则随机选择k个
            neighbors = neighbors[torch.randperm(neighbors.size(0))[:k]]
        # 在新的邻接矩阵中设置选中的邻居
        new_adj[i, neighbors] = adj[i, neighbors]
    
    return new_adj

# GCN卷积层定义
class GCNConv(nn.Module):
    def __init__(self, A, in_channels, out_channels):
        super(GCNConv, self).__init__()
        self.A_hat = A + torch.eye(A.size(0)).to(device)
        self.D = torch.diag(torch.sum(self.A_hat, 1))
        self.D = self.D.inverse().sqrt()
        self.A_hat = torch.mm(torch.mm(self.D, self.A_hat), self.D)
        
        self.W = nn.Parameter(torch.rand(in_channels, out_channels, requires_grad=True))

    def forward(self, X):
        out = torch.mm(self.A_hat,torch.mm(X, self.W))
        # out= torch.relu(out)
        return out

    def set_A(self, A):
        self.A_hat = A + torch.eye(A.size(0)).to(device)
        self.D = torch.diag(torch.sum(self.A_hat, 1))
        self.D = self.D.inverse().sqrt()
        self.A_hat = torch.mm(torch.mm(self.D, self.A_hat), self.D)


class GCN(nn.Module):
    def __init__(self, A, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(A, in_channels, hidden_channels)
        self.conv2 = GCNConv(A, hidden_channels, out_channels)
    
    def forward(self, X):
        h = self.conv1(X)
        h = torch.relu(h)
        h = self.conv2(h)
        return h

    def set_A(self, A):
        self.conv1.set_A(A)
        self.conv2.set_A(A)

# 修改模型初始化以适应节点分类
dataset = CoraGraphDataset()
g = dataset[0].to(device)
labels = g.ndata['label'].to(device)  # 获取节点标签
A_dense = g.adjacency_matrix().to_dense().to(device)
X = g.ndata['feat'].to(device)

# 定义节点分类模型
num_features = g.ndata['feat'].shape[1]
num_classes = dataset.num_classes
model = GCN(A_dense, num_features, 16, num_classes).to(device)

optimizer = optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.CrossEntropyLoss()

# 定义训练函数
def train(model, features, labels, mask):
    model.train()
    optimizer.zero_grad()
    out = model(features)
    loss = loss_fn(out[mask], labels[mask])
    loss.backward()
    optimizer.step()
    return loss.item()

# 定义评估函数
def evaluate(model, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)

# 划分训练集、验证集和测试集掩码
train_mask = g.ndata['train_mask']
val_mask = g.ndata['val_mask']
test_mask = g.ndata['test_mask']

# 训练模型
for epoch in range(200):
    loss = train(model, X, labels, train_mask)
    val_acc = evaluate(model, X, labels, val_mask)
    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss:.4f}, Val Acc: {val_acc:.4f}')

# 评估模型
model.eval()

test_acc = evaluate(model, X, labels, test_mask)
print(f'Test Accuracy: {test_acc:.4f}')

A_dense2=A_dense + torch.eye(A_dense.size(0)).to(device)
print(A_dense2)
degrees=A_dense2.sum(1)
prob=torch.zeros_like(A_dense2)
# with timer("Sampling init"):
#     for i in range(A_dense2.size(0)):
#         for j in range(A_dense2.size(1)):
#             if A_dense2[i,j]>0.5:
#                 prob[i,j]=1/degrees[i]+1/degrees[j]
#     prob/=torch.sum(prob)

mask = A_dense > 0.5  # 创建一个掩码，用于识别A_dense2中大于0.5的元素
degrees_inv = 1.0 / degrees

with timer("Sampling init"):
    prob_tmp = degrees_inv.unsqueeze(1) + degrees_inv.unsqueeze(0)  # 为什么是*不是+
    prob[mask] = prob_tmp[mask]
    prob /= prob.sum()

n_node=dataset[0].number_of_nodes()
Q=int(n_node*np.log(n_node)/(.5**2))
print(Q)

# Q=10000
Q1=Q
A_new=torch.zeros_like(A_dense2).to('cpu')
A_dense2.to('cpu')

# with timer("Sampling"):
#     for i in range(A_dense2.size(0)):
#         for j in range(A_dense2.size(1)):
#             # drawn int from bernoulli distribution of prob[i,j] and Q
#             if A_dense2[i,j]>0.5 and Q1>0:
#                 r=np.random.binomial(Q1,prob[i,j])
#                 A_new[i,j]=r/Q/prob[i,j]
#                 Q1-=r

with timer("Sampling"):
    sample_values = np.random.binomial(Q, prob[mask].to('cpu').numpy())  # 这部分保持不变，使用numpy进行采样
    print(sample_values.sum())
    sample_values_tensor = torch.from_numpy(sample_values).float()  # 将numpy数组转换为torch张量，并确保数据类型为浮点型
    
    # 进行除法运算前，确保所有参与运算的对象都是PyTorch张量
    # 注意：这里假设prob[mask]已经被正确计算并且是一个张量
    A_new[mask] = sample_values_tensor / Q / prob[mask].to('cpu')

print((A_new>0).sum())

A_new=sample_k_neighbors(A_dense, 3)

print((A_new>0).sum())

model.set_A(A_new.to(device))

# 评估模型
test_acc = evaluate(model, X, labels, test_mask)
print(f'Test Accuracy sampled: {test_acc:.4f}')