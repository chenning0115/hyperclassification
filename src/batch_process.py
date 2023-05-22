import os
import json

json_files=['./params/indian_contra_mask.json'
            # ,'./params/salinas_contra_mask.json'
            ]

# labels=[0.5,1.0]
# temps=[15,20,25,30,35,40,45,50]
# pcas=[30,35,40,45,50]
# patch_sizes=[13,15,17]
# depths=[1,2,3]
# heads=[8,16,24]

# for d in depths:
#     for h in heads:
#         for f in json_files:
#             with open(f,'r') as fin:
#                 config_in=json.load(fin)
#             config_in['net']['depth']=d
#             config_in['net']['heads']=h    
#             with open(f,'w') as fout:
#                 json.dump(config_in,fout,indent=4)
for i in range(10):
    os.system("python ./workflow.py")