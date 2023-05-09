import os
import json

json_file='./params/pavia_contra_mask.json'

dims=[128,64,32,16,8]

for dim in dims:
    with open(json_file,'r') as fin:
        config_in=json.load(fin)
    config_in['net']['dim']=dim
    with open(json_file,'w') as fout:
        json.dump(config_in,fout)
    for i in range(10):
        os.system("python ./workflow.py")