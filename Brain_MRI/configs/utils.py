import json
import re

anatomy_names = json.load(open('/DB/rhome/yichaowu/Demo_模型对接/Brain_MRI_demo/Brain_MRI/configs/anatomy_name.json','r'))

def get_anatomy_position(se):
    result = []
    replace_char = '%'
    se = se.lower()
    while 1:
        flag = False
        for ana in anatomy_names:
            for m in re.finditer(ana, se):
                flag = True
                index = m.start()
                result.append([index,index+len(ana),se[index:index+len(ana)]])
            if flag:
                se = se.replace(ana,replace_char*len(ana))
            # index = se.find(ana)
            # if index != -1:
            #     flag = True
            #     result.append([index,index+len(ana),se[index:index+len(ana)]])
            #     for idx in range(index,index+len(ana)):
            #         se[idx] = replace_char
            #     break
        if not flag:
            break
    return result

result = get_anatomy_position('The left frontal lobe and parietal lobe and left frontal lobe and frontal lobe shows high signal intensity.')

print(result)

# hammer_anas = json.load(open('/DB/rhome/yichaowu/Demo_模型对接/Brain_MRI_demo/Brain_MRI/configs/hammer_ana_names.json','r'))
# new_anas = []

# for item in hammer_anas:
#     new_anas.append(hammer_anas[item])
# new_anas += ['midline','ventricle','hippocampus','basal ganglia','temporal lobe','cerebellum','brainstem','insular lobe',
# 'occipital lobe','cingulate','frontal lobe','partial lobe','basal ganglia','thalamus','corpus callosum',"parietal lobe","left parietal lobe","right parietal lobe"]

# new_anas = list(set(new_anas))

# new_anas.sort(key = lambda x:len(x), reverse = True)

# with open('/DB/rhome/yichaowu/Demo_模型对接/Brain_MRI_demo/Brain_MRI/configs/anatomy_name.json', 'w') as f:
#     json.dump(new_anas, f, indent=4)



