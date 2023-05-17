import os
import json

json_file='./params/pavia_contra_mask.json'

patches=[13,15,17]

for patch in patches:
    with open(json_file,'r') as fin:
        config_in=json.load(fin)
    config_in['data']['patch_size']=patch
    with open(json_file,'w') as fout:
        json.dump(config_in,fout,indent=4)
    for i in range(10):
        os.system("python ./workflow.py")



# for i in range(10):
#     os.system("python ./workflow.py")