<!-- <h1><img src="figure/logo.png" alt="logo" style="height:65px; vertical-align: middle;margin-bottom: -10px;"> RaTEScore</h1> -->
<h1> UniBrain: Universal Brain MRI Diagnosis with Hierarchical Knowledge-enhanced Pre-training</h1>

Accepted by Computerized Medical Imaging and Graphics in 2025.
<div style='display:flex; gap: 0.25rem; '>
<!-- <a href='https://angelakeke.github.io/RaTEScore/'><img src='https://img.shields.io/badge/website-URL-blueviolet'></a> -->
<a href='https://www.sciencedirect.com/science/article/pii/S0895611125000254'><img src='https://img.shields.io/badge/UniBrain-Article-red'></a>
<a href='https://drive.google.com/drive/folders/1AjcxGVCGm6W40vkplXQYWmOsoG109bHH?usp=sharing'><img src='https://img.shields.io/badge/UniBrain-Model-blue'></a>

<!-- <a href='https://arxiv.org/pdf/2406.16845'><img src='https://img.shields.io/badge/Paper-PDF-red'></a> -->
</div>

## Introduction
In this study, the proposed Universal Brain MRI diagnosis system, termed as UniBrain, integrates VLP techniques into MRI-guided brain disease classification. The proposed system automatically leverages knowledge from clinical reports to provide flexible and explainable brain disease classification. Extensive experiments on both largescale clinical and public datasets demonstrate the superior diagnostic capabilities of UniBrain across more than 10 common brain diseases. Furthermore, UniBrain exhibits
robust generalization across a variety of new tasks on out-of-domain datasets. The contributions are as follows:

1. **Structured Knowledge Extraction** To extract the structured knowledge and disease labels
from free-text reports in a labor-saving manner, an Automatic Report Decomposition (ARD) module
is proposed based on the radiologist-verified clinical terms, which is open-source and compatible with
multi-center reports.
2. **Fine-grained Knowledge Alignment** To improve the efficiency of knowledge enhancement with the report, UniBrain utilizes the fine-grained report structure to improve vision feature learning at both the sequence and case levels. Such a hierarchical knowledge-enhanced pre-training scheme, applied to a large-scale clinical dataset, significantly boosts diagnostic accuracy and model scalability.
3. **System Evaluation.** UniBrain outperforms state-of-the-art open-source baselines across more than ten common brain disease types, achieving an average area under the curve (AUC) of 90.71%. Additionally, UniBrain demonstrates strong generalization capabilities, performing well on new tasks with out-of-domain datasets.

![](./assets/graphical_abstract.png)


## Model Usage

Download the UniBrain pretrained weights and text encoder weights at [here](https://drive.google.com/drive/folders/1AjcxGVCGm6W40vkplXQYWmOsoG109bHH?usp=sharing). The downloaded file should be placed at .\Brain_MRI\weights

First install the conda environment:
```shell
pip install -r requirements.txt
```

Then prepare a json file which lists information of all the testing images (nifti format please). Note that, 
- You should specify the image modality and the absolute path for each input image (T1WI, T2WI, T2FLAIR, DWI).
- The aux is the id for each modality, you can edit it at Brain_MRI\configs\modal_id.json or just use the default setting.

A possible example is shown below:
```json
[
    [
      {
        "data": "absolute/path/example/site1_065/DWI.nii.gz",
        "aux": "18"
      },
      {
        "data": "absolute/path/example/site1_065/T1WI.nii.gz",
        "aux": "1"
      },
      {
        "data": "absolute/path/example/site1_065/T2WI.nii.gz",
        "aux": "9"
      },
      {
        "data": "absolute/path/example/site1_065/T2FLAIR.nii.gz",
        "aux": "12"
      }
    ]
  ]
```
Then run the model with the following command:
```shell
python sdk_api.py
```
The output will be a list, with each element a dictionary containing the diagnosis results.
```json
[
    {
        "diagnosis": [
            "brain atrophy",
            "focal ischemia",
            "meningioma"
        ]
    }
]
```

## Contact
If you have any questions, please feel free to contact misslei@mail.ustc.edu.cn.

## Citation
```bibtex
@article{LEI2025102516,
title = {UniBrain: Universal Brain MRI diagnosis with hierarchical knowledge-enhanced pre-training},
journal = {Computerized Medical Imaging and Graphics},
pages = {102516},
year = {2025},
issn = {0895-6111},
doi = {https://doi.org/10.1016/j.compmedimag.2025.102516},
url = {https://www.sciencedirect.com/science/article/pii/S0895611125000254},
author = {Jiayu Lei and Lisong Dai and Haoyun Jiang and Chaoyi Wu and Xiaoman Zhang and Yao Zhang and Jiangchao Yao and Weidi Xie and Yanyong Zhang and Yuehua Li and Ya Zhang and Yanfeng Wang}
}
```