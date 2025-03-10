
import json
from batchgenerators.utilities.file_and_folder_operations import *

def gen_json(results,case_idx,rootdir):
    # results
    model_id = "3"
    diagnosis = ",".join(results["diagnosis"])
    anatomy_out_path = join("anatomy",model_id,case_idx)
    lesion_out_path = join("lesion",model_id,case_idx)
    icon_out_path = join("icon",model_id,case_idx)
    img_out_path = join("img",model_id,case_idx)

    img = []
    for idx in range(len(results["report"])):
        img.append({
            "report2": [{"text":item["text"],"anatomy":join(anatomy_out_path,item["anatomy"]),"lesion":join(lesion_out_path,item["lesion"])} for item in results["report"][idx]],
            "plane_id": 1,
            "aux_id": results["image"][idx]["aux_id"],
            "img_type": 3,
            "img_path": join(img_out_path,results["image"][idx]["img_name"]),
            "icon_path": join(icon_out_path,results["image"][idx]["icon_name"])
        })
    result = {
    "data":{
        "title": "Brain MRI data.",
        "system_id": 9,
        "disease_id": 1,
        "icd10_id": None,
        "certain": 1,
        "age": 20,
        "gender": 1,
        "presentation": "",
        "discussion": "",
        "component": [
            {
                "component_type": 4,
                "modality_id": 3,
                "anatomy_id": 2,
                "diagnosis1": diagnosis,
                "img": img
            }
        ]
        }
    }

    json_str = json.dumps(result, indent=4)
    with open(join(rootdir,case_idx+'.json'), 'w') as json_file:
        json_file.write(json_str)


if __name__=='__main__':
    result = {"rcd":None}
    json_str = json.dumps(result, indent=4)
    with open(join("/DB/rhome/yichaowu/Demo_模型对接/Brain_MRI_demo/data",'1.json'), 'w') as json_file:
        json_file.write(json_str)
