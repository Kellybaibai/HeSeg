import ray
import torch
import torch.nn as nn
from dgl.data import FlickrDataset, CoauthorPhysicsDataset, AmazonRatingsDataset, MinesweeperDataset, QuestionsDataset
from torch_geometric.datasets import CitationFull
from torch_geometric.data import Data
import numpy as np
import scipy.sparse as sp
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from ray.air.integrations.wandb import WandbLoggerCallback
import argparse
from scipy.stats import ttest_ind


def get_args():
    parser = argparse.ArgumentParser()
    # dataset config
    parser.add_argument('--scoring', type=str, default='AUC', help='[accuracy, AUC]')
    parser.add_argument('--threshold', type=float, default=0.6, help='the threshold of accuracy')
    parser.add_argument('--epsilon_min', type=float, default=0.3)
    parser.add_argument('--epsilon_max', type=float, default=1)
    args = parser.parse_args()
    return args



def loss_function(pos_pred, neg_pred):
    pos_loss = -torch.log(pos_pred + 1e-9).mean()
    neg_loss = -torch.log(1 - neg_pred + 1e-9).mean()
    return pos_loss + neg_loss


# GCN卷积层定义
class GCNConv(nn.Module):
    def __init__(self, A, in_channels, out_channels, device):
        super(GCNConv, self).__init__()
        self.A_hat = A
        # 计算度矩阵 D 的对角线元素，即每个节点的度
        degrees = torch.sparse.sum(self.A_hat, dim=1).to_dense()
        # 创建对角线矩阵，每个节点的度作为对角线元素
        D_values = torch.sqrt(1.0 / degrees)  # 计算度矩阵的逆矩阵的平方根的对角线元素
        D_indices = torch.LongTensor([np.arange(self.A_hat.size(0)), np.arange(self.A_hat.size(0))]) # 对角线元素的索引与节点编号相同
        print(torch.Size([self.A_hat.size(0), self.A_hat.size(0)]))
        self.D = torch.sparse.FloatTensor(D_indices.to(device), D_values,
                                            torch.Size([self.A_hat.size(0), self.A_hat.size(0)]))
        self.A_hat = torch.sparse.mm(torch.sparse.mm(self.D, self.A_hat), self.D)
        self.W = nn.Parameter(torch.rand(in_channels, out_channels, requires_grad=True))


    def forward(self, X):
        out = torch.mm(self.A_hat, torch.mm(X, self.W))
        # out= torch.relu(out)
        return out

    def set_A(self, A):
        # self.A_hat = A + torch.eye(A.size(0)).to(device)
        # self.D = torch.diag(torch.sum(self.A_hat, 1))
        # self.D = self.D.inverse().sqrt()
        self.A_hat = torch.sparse.mm(torch.sparse.mm(self.D, A), self.D)

class GAE(nn.Module):
    def __init__(self, A, in_features, hidden1, hidden2, device):
        super(GAE, self).__init__()
        # 为每个GCNConv层传递邻接矩阵A
        self.gc1 = GCNConv(A, in_features, hidden1,device)
        self.gc2 = GCNConv(A, hidden1, hidden2,device)
        self.bn = nn.BatchNorm1d(hidden1)
    
    def encode(self, X):
        h = self.gc1(X)  # 第一层GCNConv的输出
        # h = torch.relu(h)
        h = torch.pow(h, 2)
        h = self.bn(h)
        z = self.gc2(h)  # 第二层GCNConv的输出
        return z
    
    def decode(self, z, pos_edge_index, neg_edge_index):
        pos_pred = self.predict(z, pos_edge_index)
        neg_pred = self.predict(z, neg_edge_index)
        return pos_pred, neg_pred

    def predict(self, z, edge_index):
        src, dest = edge_index
        prod = torch.sum(z[src] * z[dest], dim=1)
        return torch.sigmoid(prod)
    
    def set_A(self, A):
        self.gc1.set_A(A)
        self.gc2.set_A(A)

class Data_model_preparation:
    def __init__(self, device):
        self.device = device
        # self.dataset = CoauthorCSDataset()
        # self.dataset = RomanEmpireDataset()
        self.dataset = FlickrDataset()
        # self.dataset = CoauthorPhysicsDataset()
        # self.dataset = AmazonRatingsDataset()
        # self.dataset = QuestionsDataset()
        self.avg_edges_per_node = self.dataset[0].number_of_edges() / self.dataset[0].number_of_nodes()
        print(f'Average edges per node: {self.avg_edges_per_node:.2f}')
        self.g = self.dataset[0].to(self.device)
        # self.pca_reduction()     # 是否做降维


    def get_graph_info(self):
        return self.g.number_of_nodes(), self.g.number_of_edges(), self.g.ndata['feat'].shape[1]

    def pca_reduction(self):
        # 标准化特征
        scaler = StandardScaler()
        # PCA降维
        features = self.g.ndata['feat'].cpu().numpy()   # 获取节点特征
        features_scaled = scaler.fit_transform(features)
        pca = PCA(n_components=300)  # 创建 PCA 模型
        features_pca = pca.fit_transform(features_scaled) # 将特征矩阵进行降维
        features_tensor = torch.tensor(features_pca)
        features_tensor = features_tensor.to(self.device)   # 将降维后的特征赋值给节点
        self.g.ndata['feat'] = features_tensor


    def create_model(self,learning_rate,epoch_total):
        # 数据预处理和训练测试边划分
        u, v = self.g.edges()
        eids = np.arange(self.g.number_of_edges())
        eids = np.random.permutation(eids)
        num_test = int(np.floor(eids.size * 0.1))
        train_eids = eids[num_test:]
        test_eids = eids[:num_test]
        test_pos_u, test_pos_v = u[test_eids], v[test_eids]

        # 添加负样本
        adj = sp.coo_matrix((np.ones(len(u)), (u.cpu().numpy(), v.cpu().numpy())))
        # adj_neg = 1 - adj.todense() - np.eye(self.g.number_of_nodes())
        adj_dense = adj.todense()  # 将邻接矩阵转换为稠密矩阵, shape不匹配时采用
        adj_neg = 1 - adj_dense - sp.identity(adj_dense.shape[0])  # 创建与邻接矩阵维度一致的对角矩阵
        neg_u, neg_v = np.where(adj_neg != 0)
        neg_u=torch.tensor(neg_u)
        neg_v=torch.tensor(neg_v)

        neg_eids = np.random.choice(len(neg_u), self.g.number_of_edges())
        test_neg_u, test_neg_v = neg_u[neg_eids[:num_test]], neg_v[neg_eids[:num_test]]

        # # 先将邻接矩阵转换为稀疏格式（如果是DGL图对象的话）
        adj_matrix_sparse = self.g.adjacency_matrix().to('cpu')
        adj_matrix_dense = adj_matrix_sparse.to_dense()
        A_dense = adj_matrix_dense + torch.eye(adj_matrix_dense.size(0))  # 添加自环
        A_dense = A_dense.to_sparse().to(self.device)
        X = self.g.ndata['feat'].to(self.device)
        X_mean = X.mean(dim=0, keepdim=True)
        X_std = X.std(dim=0, keepdim=True) + 1e-9
        X_norm = (X - X_mean) / X_std

        # 然后创建模型实例
        model = GAE(A_dense, self.g.ndata['feat'].shape[1], hidden1=32, hidden2=20, device=self.device).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        best_loss = float('inf')
        trigger_times = 0
        patience = 10

        for epoch in range(epoch_total):
            model.train()
            optimizer.zero_grad()
            z = model.encode(X_norm)
            pos_pred, neg_pred = model.decode(z, (u[train_eids].to(self.device), v[train_eids].to(self.device)), (neg_u[neg_eids[num_test:]].to(self.device), neg_v[neg_eids[num_test:]].to(self.device)))
            train_loss = loss_function(pos_pred, neg_pred)
            train_loss.backward()
            optimizer.step()

            # 在测试集上评估模型
            model.eval()
            with torch.no_grad():
                z = model.encode(X_norm)
                pos_pred, neg_pred = model.decode(z, (test_pos_u.to(self.device), test_pos_v.to(self.device)), (test_neg_u.to(self.device), test_neg_v.to(self.device)))
                test_loss = loss_function(pos_pred, neg_pred)

            # 更新早停逻辑
            if test_loss < best_loss:
                best_loss = test_loss
                trigger_times = 0
            else:
                trigger_times += 1

            if epoch % 10 == 0 : #or trigger_times > 0
                pass
                print(f'Epoch {epoch}')
                print(f'Epoch {epoch}, Train Loss: {train_loss.item()}, Test Loss: {test_loss.item()}')

            if trigger_times >= patience:
                print("Early stopping!")
                break

        n_node = self.g.number_of_nodes()

        return model, test_pos_v,test_pos_u, test_neg_v,test_neg_u, X_norm, adj_matrix_dense, n_node


class Evaluation:
    def __init__(self, model, test_pos_v, test_pos_u, test_neg_v, test_neg_u, X_norm, A_dense, n_node, device, args):
        self.model = model
        self.test_pos_v = test_pos_v
        self.test_pos_u = test_pos_u
        self.test_neg_v = test_neg_v
        self.test_neg_u = test_neg_u
        self.X_norm = X_norm
        self.A_dense = A_dense
        self.device = device
        self.n_node = n_node
        self.scoring = args.scoring
        self.threshold = args.threshold

    def origin_model_eval(self):
        # 使用最佳模型评估测试集的AUC
        self.model.eval()
        with torch.no_grad():
            z = self.model.encode(self.X_norm)
            pos_pred, neg_pred = self.model.decode(z, (self.test_pos_u.to(self.device), self.test_pos_v.to(self.device)), (self.test_neg_u.to(self.device), self.test_neg_v.to(self.device)))
            if self.scoring == 'accuracy':
                # 将连续数值的预测分数转换为二进制的预测标签
                binary_pred = np.where(np.concatenate([pos_pred.cpu().numpy(), neg_pred.cpu().numpy()]) > self.threshold, 1, 0)
                # 计算准确率
                score = accuracy_score(
                    np.concatenate([np.ones(pos_pred.size(0)), np.zeros(neg_pred.size(0))]), binary_pred)
                print(f'Test Acc: {score}')
            else:
                score = roc_auc_score(
                    np.concatenate([torch.ones(pos_pred.size(0)), torch.zeros(neg_pred.size(0))]),
                                    torch.cat([pos_pred, neg_pred], dim=0).cpu().numpy())
                print(f'Test AUC: {score}')

        return score

    def sample_eval(self, q_epsilon):
        # our method
        A_dense2 = self.A_dense.to('cpu') + torch.eye(self.A_dense.size(0)).to('cpu')
        # print(A_dense2)
        degrees=A_dense2.sum(1)
        prob=torch.zeros_like(A_dense2)

        mask = A_dense2 > 0.5  # 创建一个掩码，用于识别A_dense2中大于0.5的元素
        degrees_inv = 1.0 / degrees
        prob[mask] = (degrees_inv.unsqueeze(1) + degrees_inv.unsqueeze(0))[mask]
        prob = prob / prob.sum()
        if q_epsilon < 2:
            Q = int(self.n_node * np.log(self.n_node) / (q_epsilon ** 2))
        else:
            Q = q_epsilon
        A_new = torch.zeros_like(A_dense2)

        sample_values_tensor = torch.from_numpy(np.random.binomial(Q, prob[mask].numpy())).float()  # 这部分保持不变，使用numpy进行采样,将numpy数组转换为torch张量，并确保数据类型为浮点型

        # 进行除法运算前，确保所有参与运算的对象都是PyTorch张量
        # 注意：这里假设prob[mask]已经被正确计算并且是一个张量
        A_new[mask] = sample_values_tensor / Q / prob[mask]
        sampled_edges = (A_new>0).sum()
        print('sample_edge',sampled_edges)
        # 计算不包括自环且不重复的边数
        num_self_loops = (A_new.diagonal() > 0).sum() # 统计自环的数量
        print('num_self_loops', num_self_loops)
        num_unique_edges = sampled_edges - num_self_loops # 减去自环的数量

        self.model.set_A(A_new.to_sparse().to(self.device))

        # 评估模型
        with torch.no_grad():
            z = self.model.encode(self.X_norm)
            pos_pred, neg_pred = self.model.decode(z, (self.test_pos_u.to(self.device), self.test_pos_v.to(self.device)), (self.test_neg_u.to(self.device), self.test_neg_v.to(self.device)))
            if self.scoring == 'accuracy':
                # 将连续数值的预测分数转换为二进制的预测标签
                binary_pred = np.where(np.concatenate([pos_pred.cpu().numpy(), neg_pred.cpu().numpy()]) > self.threshold, 1, 0)
                # 计算准确率
                score = accuracy_score(
                    np.concatenate([np.ones(pos_pred.size(0)), np.zeros(neg_pred.size(0))]), binary_pred)
                print(f'Sampled dataset Test Acc : {score}')
            else:
                score = roc_auc_score(
                    np.concatenate([torch.ones(pos_pred.size(0)), torch.zeros(neg_pred.size(0))]),
                                    torch.cat([pos_pred, neg_pred], dim=0).cpu().numpy())
                print(f'Sampled dataset Test AUC : {score}')


        return score, num_unique_edges, int(Q)


def model_pipeline(config,q=0):
    AUC_o_list = []
    AUC_s_list = []
    for i in range(10, 110, 10):
        np.random.seed(i)
        torch.manual_seed(i)
        # 准备数据和模型
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        prepare_model = Data_model_preparation(device)
        params = prepare_model.create_model(config["learning_rate"],config["epoch"])

        # 创建评估器
        evaluator = Evaluation(*params, device)
        auc_o = evaluator.origin_model_eval()
        auc_s,_ = evaluator.sample_eval(q)
        AUC_o_list.append(auc_o)
        AUC_s_list.append(auc_s)

        # 释放显存
        del prepare_model
        del params
        del evaluator
        torch.cuda.empty_cache()

    AUC_o = np.array(AUC_o_list).mean()
    AUC_s = np.array(AUC_s_list).mean()
    result = {'AUC_o':AUC_o,'AUC_s':AUC_s}
    return result

def find_q_min_max(config, device, args):
    np.random.seed(10)
    torch.manual_seed(10)
    # 准备数据和模型
    prepare_model = Data_model_preparation(device)
    params = prepare_model.create_model(config["learning_rate"], config["epoch"])
    # 创建评估器
    evaluator = Evaluation(*params, device, args)
    q_list = []
    edge_list = []
    for i in [args.epsilon_min, args.epsilon_max]:
        temp_result = evaluator.sample_eval(i)
        q_list.append(temp_result[2])
        edge_list.append(temp_result[1])
    print('Q_min:', q_list[1])
    print('Q_max:', q_list[0])
    print('edge_min', edge_list[1])
    print('edge_max', edge_list[0])
    # 释放显存
    del prepare_model
    del params
    del evaluator
    torch.cuda.empty_cache()

    return q_list


def perform_t_test(origin_scores, sampled_scores):
    t_stat, p_value = ttest_ind(origin_scores, sampled_scores, equal_var=False)
    # 输出结果
    print(f"T-statistic: {t_stat}")
    print(f"P-value: {p_value}")

    # 判断显著性
    if p_value < 0.05:
        print("结果具有统计显著性差异")
    else:
        print("结果没有统计显著性差异")
    return p_value


def search_q(config, args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    q_min_max = find_q_min_max(config, device, args)
    origin_score_list = []
    sample_score_list = []
    sample_edge_list = []
    # # set random seed
    for i in range(10, 110, 10):
        print(f'Random Seed: {i}')
        np.random.seed(i)
        torch.manual_seed(i)
        # 准备数据和模型
        prepare_model = Data_model_preparation(device)
        graph_info = prepare_model.get_graph_info()
        params = prepare_model.create_model(config["learning_rate"],config["epoch"])
        # 创建评估器
        evaluator = Evaluation(*params, device, args)
        # 用origin方法做evaluation
        origin_score = evaluator.origin_model_eval()
        origin_score_list.append(origin_score)
        # search different Q
        # 生成指数序列
        start = q_min_max[1]
        Q_list = np.array([start])
        ad = 0
        while Q_list[-1] * (1.05 + ad)< q_min_max[0]:
            Q_list = np.append(Q_list, int(Q_list[-1] * (1.05 + ad)))
            ad += 0.005

        edge_list = []
        sample_list = []
        for j in Q_list:
            temp_result = evaluator.sample_eval(j)
            sample_list.append(temp_result[0])
            edge_list.append(temp_result[1])
        sample_score_list.append(sample_list)
        sample_edge_list.append(edge_list)

        # 释放显存
        del prepare_model
        del params
        del evaluator
        torch.cuda.empty_cache()

    origin_score_arr = np.array(origin_score_list)
    sample_score_arr = np.array(sample_score_list)

    # 做t检验
    q_value_list = []
    for cnt in range(len(Q_list)):
        print(origin_score_list)
        print(sample_score_arr[:,cnt])
        q_value_list.append(perform_t_test(origin_score_list, sample_score_arr[:,cnt]))
    # 取不同seed下的平均值
    origin_score_mean = np.mean(origin_score_arr)
    origin_score_std = np.std(origin_score_arr)
    sample_score_mean = np.mean(sample_score_arr,axis=0)
    sample_std_arr = np.std(sample_score_arr,axis=0)
    sample_edge_arr = np.mean(np.array(sample_edge_list), axis=0, dtype=int)
    sample_arr = np.array([Q_list, sample_edge_arr, sample_score_mean, sample_std_arr])
    # 将不同seed下auc的平均值保存到txt文件
    dataset_name = ['Flickr','CoraFull','AmazonComputer','CoauthorPhysics','RomanEmpire','AmazonRatings','Minesweeper','Questions']
    data_name = dataset_name[0]
    with open('./performance/he_seg_' + data_name + '.txt','w') as f:
        f.write(data_name+'\n')
        f.write('num_nodes,num_edges,num_features\n')
        f.write(','.join([str(element) for element in graph_info]) + '\n')
        f.write('Origin_'+args.scoring+',Origin_std'+'\n')
        f.write(str(origin_score_mean) + ',' + str(origin_score_std) + '\n')
        f.write('Q,sample_edges,sample_' + args.scoring + ',sample_std,p_value' + '\n')
        for row in sample_arr:
            f.write(','.join([str(element) for element in row]) + '\n')
        f.write(','.join([str(element) for element in q_value_list]) + '\n')


def search_param():
    # ray.init(object_store_memory=2 * 1024 * 1024 * 1024, ignore_reinit_error=True)
    # 定义搜索空间
    search_space = {
        "learning_rate": tune.grid_search(np.concatenate((np.arange(0.001, 0.01, 0.001),np.arange(0.01,0.15,0.01)))),
        "epoch":tune.grid_search(list(range(250,651,50)))
    }

    # 配置 Ray Tune
    analysis = tune.run(
        model_pipeline,
        config=search_space,
        resources_per_trial={"cpu": 10, "gpu": 1 if torch.cuda.is_available() else 0},
        num_samples=4,
        progress_reporter=CLIReporter(metric_columns=["learning_rate", "epoch", "AUC_o", "AUC_s", "sub"]),
        scheduler=ASHAScheduler(metric="sub", mode="min"),  # 使用ASHA调度器
        name="tune_param_experiment",
        callbacks=[WandbLoggerCallback(
            project="Pumbed_sample_1",
            log_config=True
        )]
    )

    # 获取最佳参数
    best_config = analysis.get_best_config(metric="sub", mode="min")
    best_auc = analysis.get_best_trial(metric="sub", mode="min")
    print("Best config: ", best_config, "   Best sub: ", best_auc)
    return best_config, best_auc



if __name__ == "__main__":
    args = get_args()
    search_q({"learning_rate":0.006,"epoch":300}, args)
    # best_param,best_auc = search_param()
    # print(model_pipeline({"learning_rate":0.006,"epoch":300}))
