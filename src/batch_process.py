import os
import json
from utils import check_convention, config_path_prefix
from workflow import train_by_param, train_convention_by_param

exchange_json_file='%s/temp.json' % config_path_prefix

def simple_run_times():
    # times = 5 #每个配置跑5次
    # sample_num = [10, 20, 30, 40, 50, 60, 70, 80]
    times = 1 #每个配置跑5次
    sample_num = [40]

    configs = [
        'indian_cross_param_use.json',
        'pavia_cross_param_use.json',
        'salinas_cross_param_use.json',
    ]
    for config_name in configs:
        path_param = '%s/%s' % (config_path_prefix, config_name )
        with open(path_param, 'r') as fin:
            params = json.loads(fin.read())
            data_sign = params['data']['data_sign']
            for num in sample_num:
                for t in times:
                    uniq_name = "%s_%s_%s" % (data_sign, num, t)
                    params['data']['data_file'] = '%s_%s' % (data_sign, num)
                    params['uniq_name'] = uniq_name
                    with open(exchange_json_file,'w') as fout:
                        json.dump(exchange_json_file,fout)
                    print("schedule %s..." % uniq_name)
                    os.system('python ./workflow.py')
                    print("schedule done of %s..." % uniq_name)


                
    


if __name__ == "__main__":
    simple_run_times()