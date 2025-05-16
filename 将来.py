from pyvqnet.dtype import *
from pyvqnet.tensor.tensor import QTensor
from pyvqnet.tensor import tensor
import numpy as np
from sklearn.metrics import f1_score
import pandas as pd
from pyvqnet.data.data import data_generator
from pyvqnet.nn.module import Module
from pyvqnet.nn import Linear, ReLu
from pyvqnet.optim.adam import Adam
from pyvqnet.nn.loss import CrossEntropyLoss
from pyvqnet.nn import Softmax
import pyqpanda3.core as pq
from pyqpanda3.core import QCircuit,QProg,CNOT,H,measure
from pyvqnet.qnn.pq3.measure import ProbsMeasure
from pyvqnet.qnn.pq3.quantumlayer import QuantumLayer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# 加载数据
data = pd.read_csv(r"C:\Users\34923\Desktop\american_bankruptcy.csv")

# 随机抽样平衡正负样本
# 分离正负样本
positive = data[data['status_label'] == 'alive'].sample(n=5000, random_state=42)
negative = data[data['status_label'] == 'failed'].sample(n=5000, random_state=42)

# 合并并打乱顺序
balanced_data = pd.concat([positive, negative], axis=0)
balanced_data = balanced_data.sample(frac=1, random_state=42).reset_index(drop=True)

print('\n平衡后的数据基本信息：')
balanced_data.info()

# 查看数据集行数和列数
rows, columns = balanced_data.shape

if rows < 100 and columns < 20:
    print('\n数据全部内容信息：')
    print(balanced_data.to_csv(sep='\t', na_rep='nan'))
else:
    print('\n数据前几行内容信息：')
    print(balanced_data.head().to_csv(sep='\t', na_rep='nan'))

# 提取特征和目标变量
X = balanced_data.drop(['company_name', 'status_label','year'], axis=1)
y = balanced_data['status_label']
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

print(y)#活着的是1
#加标准化
scaler = StandardScaler()
X = scaler.fit_transform(X)
# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y  # 保持数据分布一致性
)
#定义量子电路
def pqctest (input,param):
    num_of_qubits = 4

    m_machine = pq.CPUQVM()

    qubits = range(num_of_qubits)

    circuit = pq.QCircuit(4)
    for i in range(4):
        circuit << pq.RY(i, input[i])
    #print(circuit)
    for i in range(4):
        circuit << pq.RY(i, param[i])
    for i in range(3):
        circuit << pq.CNOT(i, i + 1)
    prog = QProg()
    prog << circuit
    for i in range(4):
        prog << measure(i, i)

    prog = pq.QProg()
    prog<<circuit

    rlt_prob = ProbsMeasure(m_machine,prog,[0,2])
    return rlt_prob
#pqc = QuantumLayer(pqctest,4)






# 定义神经网络模型
class Net(Module):
    def __init__(self, input_size, hidden_size):
        super(Net, self).__init__()
        self.fc1 = Linear(18, 18)
        self.relu = ReLu()
        self.fc2 = Linear(18, 18)
        self.relu = ReLu()
        self.fc3 = Linear(18, hidden_size)
        self.relu = ReLu()
        self.qc=QuantumLayer(pqctest, hidden_size)
        self.relu = ReLu()
        self.fc4 = Linear( hidden_size, 2)  # 4 个类别

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.qc(x)
        x = self.fc4(x)
        return x


# 设置模型参数
input_size = 18
hidden_size = 4  # 合理选择隐藏层神经元个数以控制参数量
model = Net(input_size, hidden_size)

# 计算参数量
total_params = 0
for param in model.parameters():
    total_params += param.numel()
print(f"模型参数量: {total_params}")

# 使用自定义损失函数
criterion = CrossEntropyLoss()  # 假设类别维度是1
optimizer = Adam(model.parameters(), lr=0.0005)

# 训练模型
num_epochs = 10
batch_size = 32

total_batches = (len(X_train) + batch_size - 1) // batch_size

for epoch in range(num_epochs):
    train_loader = data_generator(X_train, y_train, batch_size, True)
    running_loss = 0.0
    for batch_X, batch_y in train_loader:
        batch_X = QTensor(batch_X, dtype=kfloat32)
        batch_y = QTensor(batch_y, dtype=kint64)

        # 打印批次数据形状，用于调试
        #print(f"Batch X shape: {batch_X.shape}")
        #print(f"Batch y shape: {batch_y.shape}")

        optimizer.zero_grad()
        outputs = model(batch_X)
        #print(outputs)
        # 打印模型输出形状，用于调试
        #print(f"Output shape: {outputs.shape}")

        loss = criterion(batch_y, outputs)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f'Epoch {epoch + 1}, Loss: {running_loss / total_batches}')

# 在测试集上评估模型
test_X_tensor = QTensor(X_test, dtype=kfloat32)
test_y_tensor = QTensor(y_test, dtype=kint64)
# 关闭梯度计算（模拟tensor.no_grad()）
model.eval()  # 设置模型为评估模式




test_outputs = model(test_X_tensor)
#predicted = test_outputs.max(1)  # 获取最大值索引作为预测结果
#print(predicted)
# 计算准确率
layer = Softmax()
y = layer(test_outputs)
y1 = y.to_numpy()
predicted = np.argmax(y1, axis=1)
"""print(y1)
print(predicted)
print(y.shape)"""

test_y_tensor=test_y_tensor.to_numpy()


accuracy = accuracy_score(test_y_tensor, predicted)
# 计算F1分数
f1 = f1_score(test_y_tensor, predicted, average='macro')

print(f'准确率: {accuracy}')
print(f'平均 F1 分数: {f1}')

# 将结果保存到文本文件中
with open(r'D:\1python\hhh.csv', 'w') as file:
    file.write(f'准确率: {accuracy}\n')
    file.write(f'平均 F1 分数: {f1}\n')