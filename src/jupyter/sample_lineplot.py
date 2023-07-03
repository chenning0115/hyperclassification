import numpy as np
import json
# 首先要进行汇总，分别尝试使用originlab画图和matplotlib画图，使用效果较好的。（前者）
# 生成originLab可读的数据，需要汇总并将数据格式化为text文件
# 横坐标：perclass。每个模型一条线
datasets=[
    'IP',
    'PU',
    'SA'
]
models=[
    '1DCNN',
    '2DCNN',
    'SF',
    'SSRN',
    'SSFTT',
    'SSGRN',
    'DVML+SVM',
    'S3',
    'ours'
]
nums=range(10,90,10)


# np.ndarray((8,len(models)),dtype=np.float64)
for dataset in datasets:
    all_nums_data=[]
    title=['Label amount']
    title.extend(['Overall Accuracy']*len(models))
    comments=['#']
    comments.extend(models)
    all_nums_data.append(title)
    all_nums_data.append(models)
    dir=dataset
    for perclass in nums:
        path1='%s/perclass_%d/'%(dir,perclass)
        oas=[str(perclass)]
        for m in models:
            path2=path1+m
            with open(path2,'r') as fin:
                res=json.load(fin)
            oas.append(str(res['eval']['oa']*100 if res['eval']['oa']<1 else res['eval']['oa']))
        all_nums_data.append(oas)
    # 所有都统计完成，写入文档
    with open('%s/%s_OA_statistic'%(dir,dataset),'w') as fout:
        for line in all_nums_data:
            fout.write('\t'.join(line)+'\n')
    print('write %s finished'%dataset)