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

# cuda 내림차순으로 가져오기
# select * from gpu_cuda ORDER BY cuda desc, gpu desc
sql = "select * from gpu_cuda ORDER BY cuda desc, gpu desc"
cur.execute(sql)
gpu_cuda = cur.fetchall() # 0:gpu 1:cuda
con.close