import scipy.io
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import TensorDataset, DataLoader
import torchmetrics


mat_file_name = './mat/sEMG 8.mat'
mat_file = scipy.io.loadmat(mat_file_name)

emg = mat_file['emg']
label = mat_file['label']
rep = mat_file['repetition']

MAX_SIZE = 400
CHANNEL_LENGTH = 12
REPETITION = 6
CLASS_LENGTH = 17

segments = []
# traning_sets = []
x_train = np.empty((0, 36))
x_test = np.empty((0, 36))
y_train = np.array([])
y_test = np.array([])

for i in range(0, CLASS_LENGTH):
    segments.append([])
    # traning_sets.append([])
    for j in range(0, REPETITION):
        segments[i].append([])
        # traning_sets[i].append([])

# 17 * 6 * N / 400 * 400 * 12: 5차원 배열 만들기
for i, signals in enumerate(emg):
    if label[i] != 0:
        segment = segments[int(label[i]-1)][int(rep[i]-1)]
        if len(segment) == 0:
            segment.append([signals])
        elif len(segment[-1]) < MAX_SIZE:
            segment[-1].append(signals)
        else:
            segment.append([signals])


# 17 * 6 *  N // 400 * 3 * 12
x_graph= range(0, 12)
y_graph = []
for i, rep in enumerate(segments):
    for j, batch in enumerate(rep): # [ [400개씩 들어가있는 데이터] ... ]
        for k, segment in enumerate(batch): # [400개씩 들어가있는 데이터]
            # [ 12개의 채널에서 들어온 데이터]
            assert len(segment) <= 400
            mav = np.mean(np.abs(segment), axis=0)
            # assert mav.shape == (12,)
            var = np.var(segment, axis=0)
            # assert var.shape == (12,)
            wl = np.sum(np.abs(np.subtract(segment[:-1], segment[1:])), axis=0)
            # segment.shape == (BATCH, 12)
            # segment = [b0, b1, b2, b3, b4, ... b399]
            #      b0 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
            #      b1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
            if i == 0 and j == 0:
                y_graph.append(mav)
            if j < 4:
                x_train = np.append(x_train, np.array([mav, var, wl]).reshape((1, 36)), axis=0)
                y_train = np.append(y_train, np.full(1, i))
            else:
                x_test = np.append(x_test, np.array([mav, var, wl]).reshape((1, 36)), axis=0)
                y_test = np.append(y_test, np.full(1, i))
            # traning_sets[i][j].append([mav, var, wl])

print('x_train', x_train.shape)
print('y_train', y_train.shape)
print('x_test', x_test.shape)
print('y_test', y_test.shape)

# 그래프 그리기
plt.title('L1-1 MAV')
plt.xlabel('channel')
plt.xticks(x_graph)
plt.ylabel('MAV')
plt.legend(loc='upper right')
plt.plot(x_graph, np.transpose(np.array(y_graph)), label='rep1')
plt.show()

device = torch.device("mps") if torch.backends.mps.is_available() else "cpu"
print('device', device)
# 랜덤 시드 고정
torch.manual_seed(777)

# GPU 사용 가능일 경우 랜덤 시드 고정
if device == 'cuda':
    torch.cuda.manual_seed_all(777)

learning_rate = 0.001
training_epochs = 100000
batch_size = 775

model = nn.Sequential(
    nn.Linear(36, 1024),
    nn.LeakyReLU(),
    nn.Linear(1024, 1024), 
    nn.LeakyReLU(),
    nn.Linear(1024, 1024),
    nn.ReLU(),
    nn.Linear(1024, 17),
)

ds_train = TensorDataset(torch.Tensor(x_train), torch.LongTensor(y_train))
ds_test = TensorDataset(torch.Tensor(x_test), torch.LongTensor(y_test))

loader_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
loader_test = DataLoader(ds_test, batch_size=batch_size, shuffle=False)

loss_fn = nn.CrossEntropyLoss() # 이 비용 함수는 소프트맥스 함수를 포함하고 있음.
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
losses = []

confmat = torchmetrics.ConfusionMatrix(task="multiclass", num_classes=CLASS_LENGTH)

def train(epoch):
    model.train()  # 신경망을 학습 모드로 전환

    # 데이터로더에서 미니배치를 하나씩 꺼내 학습을 수행
    for data, targets in loader_train:

        optimizer.zero_grad()  # 경사를 0으로 초기화
        outputs = model(data)  # 데이터를 입력하고 출력을 계산
        loss = loss_fn(outputs, targets)  # 출력과 훈련 데이터 정답 간의 오차를 계산
        loss.backward()  # 오차를 역전파 계산
        optimizer.step()  # 역전파 계산한 값으로 가중치를 수정

    print("epoch{}：완료\n".format(epoch))

def test():
    model.eval()  # 신경망을 추론 모드로 전환
    correct = 0

    # 데이터로더에서 미니배치를 하나씩 꺼내 추론을 수행
    with torch.no_grad():  # 추론 과정에는 미분이 필요없음
        for data, targets in loader_test:
            outputs = model(data)  # 데이터를 입력하고 출력을 계산
        

            # 추론 계산
            _, predicted = torch.max(outputs.data, 1)  # 확률이 가장 높은 레이블이 무엇인지 계산
            correct += predicted.eq(targets.data.view_as(predicted)).sum()  # 정답과 일치한 경우 정답 카운트를 증가
            print('confmat', np.matrix(confmat(predicted, targets)))

    # 정확도 출력
    data_num = len(y_test)  # 데이터 총 건수
    print('\n테스트 데이터에서 예측 정확도: {}/{} ({:.0f}%)\n'.format(correct,
                                                   data_num, 100. * correct / data_num))
    
    return correct

corrects = []
for i in range(1, 11):
    batch_size = 2 ** i
    train(training_epochs)
    correct = test()
    corrects.append(correct / len(y_test))
    print(batch_size)

# 그래프 출력
x = list(map(lambda x: 2 ** x, range(1, 11)))
plt.plot(x, corrects, marker='o')
plt.xticks(x, rotation=90, fontsize=5)
plt.show()

train(training_epochs)
test()