from Brain_MRI import inferenceSdk as InferenceSdk # 仅导入包，不涉及任何模型初始化的工作
import json

import time

# s = time.time()
model = InferenceSdk.RatiocinationSdk(gpu_id=[1], inference_cfg='Brain_MRI/configs/config_UniBrain.yaml') # 模型初始化（加载权重等）gpu_id: 显卡号,
# e = time.time()

# print("~~~~~~~~!!!!!loading model",e-s,"seconds")

input_case_dict = json.load(open('/DB/rhome/yichaowu/Demo_模型对接/Brain_MRI_demo/example/input_official_4.json','r'))

results = model.diagRG(input_case_dict)

with open('/DB/rhome/yichaowu/Demo_模型对接/Brain_MRI_demo/example/output_official_4.json', 'w') as f:
    json.dump(results, f, indent=4)