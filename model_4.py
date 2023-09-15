# %% 
# 1. 기본 데이터 불러오기
from traceback import print_tb
from numpy import average
import pymysql

con = pymysql.connect(host='localhost', port=8889, user='root', password='root', db='data_science', charset='utf8')

cur = con.cursor()

filter_cpu = [
    "Intel Core i5-10400 Processo",
    "Intel Core i5-10400F Processor",
    "Intel Core i5-10500 Processor",
    "Intel Core i5-10600K Processor",
    "Intel Core i5-10600KF Processor",
    "Intel Core i5-11400 Processor",
    "Intel Core i5-11400F Processor",
    "Intel Core i5-11600K Processor",
    "Intel Core i5-12400 Processor",
    "Intel Core i5-12400F Processor",
    "Intel Core i5-12500 Processor",
    "Intel Core i5-12600K Processor",
    "Intel Core i5-12600KF Processor",
    "Intel Core i5-8400 Processor",
    "Intel Core i5-8500 Processor",
    "Intel Core i5-8600 Processor",
    "Intel Core i5-8600K Processor",
    "Intel Core i5-9400 Processor",
    "Intel Core i5-9400F Processor",
    "Intel Core i5-9500 Processor",
    "Intel Core i5-9500F Processor",
    "Intel Core i5-9600K Processor",
    "Intel Core i5-9600KF Processor",
    "Intel Core i7-10700K Processor",
    "Intel Core i7-10700KF Processor",
    "Intel Core i7-10700 Processor",
    "Intel Core i7-10700F Processor",
    "Intel Core i7-11700K Processor",
    "Intel Core i7-11700KF Processor",
    "Intel Core i7-11700 Processor",
    "Intel Core i7-11700F Processor",
    "Intel Core i7-12700F Processor",
    "Intel Core i7-12700K Processor",
    "Intel Core i7-12700KF Processor",
    "Intel Core i7-8700K Processor",
    "Intel Core i7-8700 Processor",
    "Intel Core i7-9700K Processor",
    "Intel Core i7-9700KF Processor",
    "Intel Core i7-9700F Processor",
    "Intel Core i7-9700 Processor",
    "Intel Core i9-10850K Processor",
    "Intel Core i9-10900K Processor",
    "Intel Core i9-10900KF Processor",
    "Intel Core i9-10900 Processor",
    "Intel Core i9-10900F Processor",
    "Intel Core i9-11900K Processor",
    "Intel Core i9-11900KF Processor",
    "Intel Core i9-11900 Processor",
    "Intel Core i9-12900KF Processor",
    "Intel Core i9-12900K Processor"
]

# benchmark_score db에서 전부 가져와서 배열화 (상위 필터 조건에 해당하는 데이터만)
sql_where = " where"

for cpu in filter_cpu:
    sql_where = sql_where + " cpu='" + cpu + "' OR"
sql_where = sql_where[:-3]

sql = "SELECT * FROM benchmark_score"
cur.execute(sql+sql_where)
benchmark_score = cur.fetchall() # 0:num 1:rank 2:overallScore 3:graphicScore 4:cpuScore 5:cpu 6:gpu
con.close

# 각 그래픽 카드 별 평균 점수
count_gpu = {}
sum_gpu_score = {}
average_gpu_score = {}
for value in benchmark_score:
    # 그래픽 카드 별로 점수 전부 더해주기
    gpu_name = value[6]
    gpu_score = value[3]
    if gpu_name in sum_gpu_score: # value[6] 그래픽카드 이름이 이미 sum_gpu_score 에 존재한다면
        sum_gpu_score[gpu_name] = sum_gpu_score[gpu_name] + gpu_score
        count_gpu[gpu_name] = count_gpu[gpu_name] + 1
    else:
        sum_gpu_score[gpu_name] = gpu_score
        count_gpu[gpu_name] = 1

# 점수 평균 구하기
for key in sum_gpu_score:
    average_gpu_score[key] = sum_gpu_score[key] / count_gpu[key]

# %% 
# 2. 모델별 cuda 불러오기

# cuda 내림차순으로 가져오기
sql = "select * from gpu_cuda ORDER BY cuda desc, gpu desc"
cur.execute(sql)
gpu_cuda = cur.fetchall() # 0:gpu 1:cuda
con.close

# %%
# 30 세대 그래픽카드의 CUDA 조정
gpu_cuda = list(gpu_cuda)
for index in range(0, len(gpu_cuda)):
    if 'NVIDIA GeForce RTX 30' in gpu_cuda[index][0]:
        changed_cuda = gpu_cuda[index][1]*0.75
        gpu_cuda[index] = (gpu_cuda[index][0], changed_cuda)

gpu_cuda.sort(key=lambda x:(x[1], x[0]), reverse=True)
gpu_cuda = tuple(gpu_cuda)

# GTX 모델 그래픽카드의 CUDA 조정
gpu_cuda = list(gpu_cuda)
for index in range(0, len(gpu_cuda)):
    if 'NVIDIA GeForce GTX' in gpu_cuda[index][0]:
        changed_cuda = (gpu_cuda[index][1]/136)*100
        gpu_cuda[index] = (gpu_cuda[index][0], changed_cuda)

gpu_cuda.sort(key=lambda x:(x[1], x[0]), reverse=True)
gpu_cuda = tuple(gpu_cuda)

# %%
# gpu_cuda 를 3d - cuda 내림차순으로 정렬
gpu_cuda = list(gpu_cuda)
for index in range(0, len(gpu_cuda)):
    temp_score_minus_cuda = average_gpu_score[gpu_cuda[index][0]] - gpu_cuda[index][1]
    gpu_cuda[index] = (gpu_cuda[index][0], gpu_cuda[index][1], temp_score_minus_cuda)

gpu_cuda.sort(key=lambda x:(x[2], x[0]), reverse=True)
gpu_cuda = tuple(gpu_cuda)

#%%
# gpu 랑 cuda 배열화
gpu = []
cuda = []
for value in gpu_cuda:
    gpu.append(value[0])
    cuda.append(value[1])

# 3d-cuda
score_minus_cuda = []
for value in gpu_cuda:
    result = average_gpu_score[value[0]] - value[1]
    score_minus_cuda.append(result)

# %% 3d-cuda 그리기
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(8,6), dpi=144)
ax = fig.add_subplot(111, frame_on=False) 
data = []
for value in gpu_cuda:
    data.append([value[0], value[1], average_gpu_score[value[0]]])
column_labels=["gpu", "cuda", "3DMark"]
ax.axis('tight')
ax.axis('off')
ax.table(cellText=data,colLabels=column_labels,loc="center")

plt.show()

plt.plot(score_minus_cuda)
plt.title("Score - Cuda")
plt.show()

# %% 각 꼭짓점 기울기의 전체 평균
dxdy = []
for index in range(0, len(score_minus_cuda)-1):
    dxdy.append(score_minus_cuda[index] - score_minus_cuda[index+1])
average_dxdy = average(dxdy)

print("average_dxdy : " + str(average_dxdy))

# %% 식의 P값 구하기
p_list = []
p_index = 0
for rank in range(28, 0, -1):
    result = score_minus_cuda[p_index] - average_dxdy * rank
    p_index += 1
    p_list.append(result)
    print(result)

# %% 실제 Score와 예측 Score의 정확도가 가장 좋게 나오게 하는 P값

# 예측 3DMark 구하기
fact_score = []
for value in gpu_cuda:
    fact_score.append(average_gpu_score[value[0]])

sum_abs_minus = []
rank = range(28, -1, -1)
p_value = []
for index in range(0, 29):
    p_value.append((fact_score[index]-cuda[index])-average_dxdy*rank[index])

for p in p_value:
    # Y 값 구해주기
    y = []
    for index1 in range(0,29):
        y.append(average_dxdy*rank[index1]+p)
    # 예측 3DMARK 값
    predict_score = []
    for index2 in range(0,29):
        predict_score.append(y[index2]+cuda[index2])
    # adb(예측-실제) 구해서 배열
    abs_minus = []
    for index3 in range(0,29):
        abs_minus.append(abs(predict_score[index3]-fact_score[index3]))
    sum_abs_minus.append(sum(abs_minus))

best_p_index = sum_abs_minus.index(min(sum_abs_minus))
print(str(best_p_index) + "번째")
print("가장 좋은 P값 : " + str(p_value[best_p_index]))
p = p_value[best_p_index]


# %% 정확도 구하기
accuracy = sum(abs_minus)
accuracy_percent = []
for index in range(0, len(fact_score)):
    result = 100 - abs(100-(100*(fact_score[index]+predict_score[index])/(fact_score[index]*2)))
    print(result)
    accuracy_percent.append(result)
average_accuracy = average(accuracy_percent)
print("정확도율 : " + str(average_accuracy))

# %% 그래프 그리기
plt.plot(fact_score, 'r')
plt.plot(predict_score, 'b')
plt.show

# %% 40세대 예측하기 (40세대 또한 0.75 곱셈)
gpu_cuda_with_40 = list(gpu_cuda)

gpu_cuda_with_40.append(('NVIDIA GeForce RTX 4090', 18432*0.75))
gpu_cuda_with_40.append(('NVIDIA GeForce RTX 4080', 10752*0.75))
gpu_cuda_with_40.append(('NVIDIA GeForce RTX 4070', 7680*0.75))
gpu_cuda_with_40.append(('NVIDIA GeForce RTX 4060', 4608*0.75))
gpu_cuda_with_40.append(('NVIDIA GeForce RTX 4050', 3072*0.75))

gpu_cuda_with_40.sort(key=lambda x:(x[1], x[0]), reverse=True)

gpu_cuda_with_40 = tuple(gpu_cuda_with_40)

predict_score_with_40 = []
rank = list(range(len(gpu_cuda_with_40)-1, -1, -1))

for index in range(0, len(gpu_cuda_with_40)):
    result = (average_dxdy*rank[index] + p) + gpu_cuda_with_40[index][1]
    predict_score_with_40.append(result)
    print(result)

# %% 최종 예측 데이터 표
fig = plt.figure(figsize=(8,6), dpi=144)
ax = fig.add_subplot(111, frame_on=False)
data = []
for index in range(0, len(gpu_cuda_with_40)):
    data.append([gpu_cuda_with_40[index][0], predict_score_with_40[index]])
column_labels=["gpu", "predict score"]
ax.axis('tight')
ax.axis('off')
ax.table(cellText=data,colLabels=column_labels,loc="center")

plt.show()
# %%
