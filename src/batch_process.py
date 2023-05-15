import os
import json

json_file='./params/pavia_contra_mask.json'

# labels=[0.5,1.0]
# temps=[15,20,25,30,35,40,45,50]

# for l in labels:
#     for t in temps:
#         with open(json_file,'r') as fin:
#             config_in=json.load(fin)
#         config_in['data']['unlabelled_multiple']=l
#         config_in['train']['temp']=t
#         with open(json_file,'w') as fout:
#             json.dump(config_in,fout,indent=4)
for i in range(10):
    os.system("python ./workflow.py")