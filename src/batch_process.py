import os
import json

json_files=['./params/indian_contra_mask.json'
            ,'./params/salinas_contra_mask.json'
            ,'./params/houston_contra_mask.json'
            # ,'./params/pavia_contra_mask.json'
            ]

# labels=[0.5,1.0]
# temps=[15,20,25,30,35,40,45,50]
# pcas=[30,35,40,45,50]
# patch_sizes=[13,15,17]
# depths=[1,2,3]
# heads=[8,16,24]

for perclass in range(10,90,10):
    for s in range(1,4):
        for f in json_files:
            with open(f,'r') as fin:
                config_in=json.load(fin)
            config_in['data']['perclass']=perclass
            config_in['data']['sample']=s
            with open(f,'w') as fout:
                json.dump(config_in,fout,indent=4)
        for i in range(10):
            os.system("python ./workflow.py")

# 删除作废文件
# dir='./res/one_step'
# for s in ['Houston','Salinas','Indian']:
#     for perclass in range(10,90,10):
#         for samp in range(1,4):
#             path='%s/%s/%d/%d/'%(dir,s,perclass,samp)
#             files=os.listdir(path)
#             for f in files:
#                 os.remove(path+'/'+f)