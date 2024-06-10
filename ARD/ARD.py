# -*- encoding: utf8 -*-

import pandas as pd
from random import choice
import numpy as np
import json
from deep_translator import GoogleTranslator

class ARD:
    def __init__(self, keyword_path) -> None:
        self.keywords, self.keytrans = self.extract_keywords(keyword_path)
   
    def process_disease(self,cur_dis):
        report = cur_dis
        res = []
        last = report.find("附件")
        if last!=-1:
            report = report[:last].strip()
        last = report.find("附见")
        if last != -1:
            report = report[:last].strip()
        groups = report.split('。')
        for group in groups:

            origin_temp = group
            origin_idxs = [idx for idx in range(len(group))]
            extract = []
            for m in ["disease","anatomy","dura","scalp","ignore"]:
                cnt = 0
                temp_report = origin_temp
                idxs = origin_idxs
                while cnt != len(self.keywords[m]):
                    cnt = 0
                    for i in self.keywords[m]:
                        start = temp_report.find(i)
                        if start!=-1:
                            if m != "ignore":
                                extract.append([idxs[start],m,i])
                            end = start+len(i)
                            temp_report = temp_report[:start]+temp_report[end:]
                            idxs = idxs[:start]+idxs[end:]
                            cnt = 0
                        else:
                            cnt += 1
            extract.sort(key=lambda k: k[0])

            ### process hema and dura ###

            for idx,cur in enumerate(extract):
                ana_type, ana_cont = cur[1], cur[2]
                if ana_type == "scalp":
                    disease_index = []
                    for follow in range(idx+1,len(extract)):
                        if extract[follow][1] == "anatomy":
                            break
                        if extract[follow][1] == "disease":
                            disease_index.append(follow)
                    if len(disease_index) == 0:
                        extract[idx][1]="disease"
                        extract[idx][2]="忽略"
                    else:
                        for theidx in disease_index:
                            extract[theidx][2] = "忽略"
                elif ana_type == "dura":
                    has_anatomy = False
                    hema_index = []
                    self_trans = False
                    self_trans_maybe = False
                    maybe_index = []
                    for follow in range(idx+1,len(extract)):
                        if extract[follow][1] == "anatomy":
                            has_anatomy = True
                        elif extract[follow][1] == "disease": 
                            if extract[follow][2] == "积液" or self.keytrans['disease'][extract[follow][2]] == "maybe hematencephalon":
                                if not has_anatomy:
                                    maybe_index.append(follow)
                                else:
                                    self_trans_maybe = True
                            elif self.keytrans['disease'][extract[follow][2]] == "hematencephalon":
                                if not has_anatomy:
                                    hema_index.append(follow)
                                else:
                                    self_trans = True
                            else:
                                break
                    if len(maybe_index):
                        for theidx in maybe_index:
                            extract[theidx][1] = "disease"
                            extract[theidx][2] = "可能硬膜血肿"
                    elif self_trans_maybe:
                        extract[idx][1]="disease"
                        extract[idx][2]="可能硬膜血肿"
                    
                    if len(hema_index):
                        for theidx in hema_index:
                            extract[theidx][2] = "硬膜血肿"
                    elif self_trans:
                        extract[idx][1]="disease"
                        extract[idx][2]="硬膜血肿"
            ### ###
            extract = list(filter(lambda x:x[1]=="anatomy" or x[1]=="disease", extract))
            
            for idx,cur in enumerate(extract):
                ana_type, ana_cont = cur[1], cur[2]
                ana_cont = self.keytrans[ana_type][ana_cont]
                if idx == 0:
                    res.append({ana_type:[ana_cont]})
                elif ana_type == "anatomy":
                    if extract[idx-1][1]=="disease":
                        res.append({'anatomy':[ana_cont],'disease':[]})
                    else:
                        res[-1]['anatomy'].append(ana_cont)
                elif ana_type == "disease":
                    if "disease" not in res[-1]:
                        res[-1]["disease"]=[ana_cont]
                    else:
                        res[-1]["disease"].append(ana_cont)
        return res
       
    def postprocess_report(self, report, no_sig_info):
        res = {}
        sig_label = []
        model_isointensity = {modal:False for modal in ["DWI","T2FLAIR","T1WI","T2WI"]}
        for i in report:
            if len(i['signal'])==0: # If not any signal then ignore
                continue
            modal_sig = {}
            modal_shape = {}
            for idx,m in enumerate(i['modality']):
                trans_m = self.keytrans['modality'][m]
                trans_sig = self.keytrans['signal'][i["signal"][idx]]
                trans_shape = self.keytrans['shape'][i["shape"][idx]]
                trans_sig = trans_sig+" ring" if ("ring" not in trans_sig and trans_shape == "ring") and ('center' not in trans_sig) else trans_sig
                trans_sig = trans_sig+" center" if ("center" not in trans_sig and trans_shape == "center") and ('ring' not in trans_sig) else trans_sig
                trans_shape = self.keytrans['shape'][-1] if "ring" in trans_shape or "center" in trans_shape else trans_shape
                if trans_m not in modal_sig:
                    modal_sig[trans_m] = []
                    modal_shape[trans_m] = []
                if i["signal"][idx] == -1:
                    model_isointensity[trans_m] = True
                else:
                    modal_shape[trans_m].append(trans_shape)
                    modal_sig[trans_m].append(trans_sig)
            for trans_m in modal_sig:
                if trans_m == 'ADC':
                    continue
                if trans_m not in res:
                    res[trans_m] = []
                if len(modal_sig[trans_m])>1:
                    o_former = modal_sig[trans_m][0].strip()
                    o_latter = modal_sig[trans_m][1].strip()
                    if o_former.startswith("hyperintensity") and o_latter.startswith("hyperintensity"):
                        modal_sig[trans_m] = ["hyperintensity"] + modal_sig[trans_m][2:]
                        modal_shape[trans_m] = modal_shape[trans_m][0:1]+ modal_shape[trans_m][2:]
                    elif o_former.startswith("hypointensity") and o_latter.startswith("hypointensity"):
                        modal_sig[trans_m] = ["hypointensity"] + modal_sig[trans_m][2:]
                        modal_shape[trans_m] = modal_shape[trans_m][0:1]+ modal_shape[trans_m][2:]
                    else:
                        if 'center' in o_latter or 'ring' in o_former:       
                            former = o_latter if 'center' in o_latter else o_latter+' center'
                            latter = o_former if 'ring' in o_former else o_former+' ring'
                            modal_shape[trans_m] = modal_shape[trans_m][1:2] + modal_shape[trans_m][0:1] + modal_shape[trans_m][2:]
                        else:
                            former = o_former
                            latter = o_latter
                        former = former.replace('center','').strip()
                        latter = latter.replace('ring','').strip() if latter!="hyperintensity ring" and latter!="hypointensity ring" else latter
                        modal_sig[trans_m] = [former,latter]+modal_sig[trans_m][2:]
                        # check = modal_sig[trans_m][0:1]+modal_sig[trans_m][2:]
                        check = list(filter(lambda x:"ring" not in x, modal_sig[trans_m]))
                        ring_sig = list(filter(lambda x:"ring" in x, modal_sig[trans_m]))
                        if ('hyperintensity' in check and "hypointensity" in check) or 'heterogeneous intensity' in check:
                            modal_sig[trans_m] = ['heterogeneous intensity']+ring_sig
                            modal_shape[trans_m] = modal_shape[trans_m][0:1] + ['unspecified'] if len(ring_sig) else modal_shape[trans_m][0:1]
                        else:
                            pass
                elif len(modal_sig[trans_m]) == 1:
                    if 'center' in modal_sig[trans_m][0]:
                        modal_sig[trans_m][0] = modal_sig[trans_m][0].replace('center','').strip()
                    if 'ring' in modal_sig[trans_m][0] and modal_sig[trans_m][0]!="hyperintensity ring" and modal_sig[trans_m][0]!="hypointensity ring":
                        modal_sig[trans_m][0] = modal_sig[trans_m][0].replace('ring','').strip()
                    if modal_sig[trans_m][0] == "isointensity":
                        model_isointensity[trans_m] = True
                modal_sig[trans_m] = list(map(lambda x: x.replace('center','').strip(), modal_sig[trans_m]))
                for index,trans_sig in enumerate(modal_sig[trans_m]):
                    for idx,ana in enumerate(i['anatomy']):
                        trans_ana = self.keytrans['anatomy'][ana]
                        trans_side = self.keytrans['side'][i['side'][idx]]
                        trans_shape = modal_shape[trans_m][index] if modal_shape[trans_m][index]!="unspecified" else ""
                        if 'and' in trans_ana and trans_ana!='pineal gland': # if the anatomies consist of multiple sub-anatomise by 'and'
                            split_trans_ana = trans_ana.split('and')
                            for split_an in split_trans_ana:
                                # morph+modality+signal+side+anatomy
                                res[trans_m] += [' '.join([trans_shape,trans_m,trans_sig,'on',trans_side,split_an.strip()]).strip()]
                                sig_label.append({'anatomy':split_an.strip(),'disease':trans_m+' '+trans_sig})
                        else:
                            res[trans_m] += [' '.join([trans_shape,trans_m,trans_sig,'on',trans_side,trans_ana]).strip()]
                            sig_label.append({'anatomy':trans_ana,'disease':trans_m+' '+trans_sig})
        
        trans_sig_dic = [self.keytrans['signal'][sig] for sig in ['低','高','高低混杂','等高','等低','低信号环','高信号环','等信号']]
        normal_modal = []
        for modal in ["DWI","T2FLAIR","T1WI","T2WI"]:
            if modal not in res:
                res[modal] = ["unspecified"]
                sig_label += [{'anatomy':'','disease':'maybe '+modal+' '+sig} for sig in trans_sig_dic]
        for modal in model_isointensity:
            if model_isointensity[modal]:
                if len(res[modal])==0:
                    res[modal] = ["isointensity"]
                    normal_modal.append(modal)
        
        for modal in normal_modal:
            sig_label = list(filter(lambda x:not x['disease'].startswith(modal),sig_label))
        
        # add non signal information to the modal-wise description
        # diseases = list(set([p['disease'] for p in self.reports[fid]['disease']]))
        diseases = [self.keytrans['disease']['脑积水']]
        diseases = []
        if self.keytrans['disease']['脑积水'] in diseases:
            no_sig_info[self.keytrans['no_sig_shape']['扩张明显']] += [self.keytrans['anatomy']['脑池'],self.keytrans['anatomy']['脑室']]
        
        for shape in no_sig_info:
            cur_ana = []
            for ana in no_sig_info[shape]:
                if 'and' in ana and ana != 'pineal gland':
                    for sub_ana in ana.split('and'):
                        cur_ana.append(sub_ana.strip())
                else:
                    cur_ana.append(ana)
            no_sig_info[shape] = cur_ana

        no_sig_info[self.keytrans['no_sig_shape']['扩张']] = list(set(no_sig_info[self.keytrans['no_sig_shape']['扩张']]).difference(set(no_sig_info[self.keytrans['no_sig_shape']['扩张明显']])))
        no_sig_info_text = []
        no_sig_info_label = []

        for shape in no_sig_info:
            if shape == self.keytrans['no_sig_shape']['增宽'] and (self.keytrans['anatomy']['脑沟'] in no_sig_info[shape] or self.keytrans['anatomy']['脑回'] in no_sig_info[shape]):
                no_sig_info[shape] += [self.keytrans['anatomy']['脑沟'], self.keytrans['anatomy']['脑回']]
            no_sig_info[shape] = list(set(no_sig_info[shape]))
            if len(no_sig_info[shape]) != 0 and shape != self.keytrans['no_sig_shape']['居中'] and shape != self.keytrans['no_sig_shape']['清晰']:
                for ana in no_sig_info[shape]:
                    if shape in list(self.keytrans['disease'].values()):
                        no_sig_info_text.append(shape+" is located on "+ana)
                    else:
                        no_sig_info_text.append(ana+" "+shape)
                    no_sig_info_label.append({'anatomy':ana,"disease":shape})
        
        for modal in res:
            res[modal] += no_sig_info_text

        return res

    def process_report(self, cur_report):
        report = cur_report
        extract, res = [], []

        no_sig_info = {self.keytrans['no_sig_shape'][i]:[] for i in self.keytrans['no_sig_shape']}
        no_sig_info.update({self.keytrans['disease'][i]:[] for i in self.keytrans['disease']})
        
        last = report.find("附件")
        if last!=-1:
            report = report[:last].strip()
        last = report.find("附见")
        if last != -1:
            report = report[:last].strip()
        
        # Check the type in order：anatomy, side, shape, modality, deny, signal
        # while checking each type, the words of his type is checked from the longest to the shortest
        origin_temp = report
        origin_idxs = [idx for idx in range(len(report))]
        for m in ["anatomy","shape","side","modality","deny","signal","parse","no_sig_shape","disease"]:
            cnt = 0
            temp_report = origin_temp
            idxs = origin_idxs
            while cnt != len(self.keywords[m]):
                cnt = 0
                for i in self.keywords[m]:
                    start = temp_report.find(i)
                    if start == -1:
                        cnt+=1
                    while start!=-1:
                        extract.append([idxs[start],m,i])
                        end = start+len(i)
                        temp_report = temp_report[:start]+temp_report[end:]
                        idxs = idxs[:start]+idxs[end:]
                        cnt = 0
                        if m == "anatomy":
                            origin_temp = origin_temp[:start]+origin_temp[end:]
                            origin_idxs = origin_idxs[:start]+origin_idxs[end:]
                        start = temp_report.find(i)
        extract.sort(key=lambda k: k[0])

        ana_index = [] # Group anatomies, anatomies with the same signal description is grouped together
        start=True
        for idx,i in enumerate(extract):
            if i[0] == "anatomy":
                # print(start,i[1])
                if start:
                    if idx>0 and extract[idx-1][1] == "side":
                        if extract[idx-1][0]+len(extract[idx-1][2]) >= extract[idx][0]: # If the side is on the left of this anatomy
                            ana_index.append([idx-1]) # the word of type "side" is added to the group of thie anatomy
                        else:
                            parse_flag = False
                            for index in range(extract[idx-1][0]+1,extract[idx][0]):
                                if report[index] in ["，","。","；","、"]:
                                    ana_index.append([idx])
                                    parse_flag = True
                            if not parse_flag:
                                ana_index.append([idx-1])
                    else:
                        ana_index.append([idx])
                    start = False
                else:
                    ana_index[-1].append(idx)
            elif i[0] != "side" and i[0]!=self.keywords["special_shape"]:
                start = True

        groups = [[0,-1]]
        for idx in range(1,len(ana_index)):
            groups[-1][-1]=ana_index[idx][0]
            groups.append([ana_index[idx][0],-1])
        
        for begin,end in groups:
            cur_group = extract[begin:end] if end!=-1 else extract[begin:]
            #print("cur_group1",cur_group)
            for idx,i in enumerate(cur_group):
                if i[1] == "signal": 
                    if idx>0 and cur_group[idx-1][1]=="deny":
                        i[2]=-1
                        cur_group[idx-1][1]=="ignore"
                elif i[1] == "no_sig_shape": 
                    if idx>0 and cur_group[idx-1][1]=="deny":
                        i[2]=-1
                        cur_group[idx-1][1]=="ignore"
            for idx,i in enumerate(cur_group):
                if i[1]=="deny":
                    if idx<len(cur_group)-1 and cur_group[idx+1][1]=="shape":
                        cur_group[idx+1][1]="ignore"
                    else:
                        i[1] = "ignore"

            cur_group = list(filter(lambda x:x[1]!="ignore", cur_group))

            new_cur_group = []
            right_neighbor = -1
            for idx in range(len(cur_group)-1,-1,-1):
                i = cur_group[idx]
                new_cur_group.append([i[0],i[1],i[2],-1,[],0,right_neighbor])
                right_neighbor = i[1]
            
            new_cur_group = new_cur_group[::-1]

            
            for idx in range(len(new_cur_group)-1,-1,-1):
                obj = new_cur_group[idx]
                if obj[1]=="anatomy":
                    right = False
                    for follow in range(idx+1,len(new_cur_group)):
                        cur_obj = new_cur_group[follow] 
                        if cur_obj[1] == "parse":
                            break
                        if cur_obj[1] == "side" and cur_obj[-2] == 1:
                            break
                        if cur_obj[1] != "side" and cur_obj[1] != "anatomy":
                            break
                        if cur_obj[1] == "side":
                            obj[3] = cur_obj[2]
                            right = True
                    if not right:
                        for follow in range(idx-1,-1,-1):
                            cur_obj = new_cur_group[follow]
                            if cur_obj[1] == "parse" or cur_obj[1]=="signal" or cur_obj[1]=="modality":
                                break
                            if cur_obj[1] == "side":
                                if cur_obj[0]+len(cur_obj[2]) >= obj[0]:
                                    cur_obj[-2] = 1
                                obj[3] = cur_obj[2]
                                break
                    for follow in range(idx+1,len(new_cur_group)):
                        cur_obj = new_cur_group[follow]
                        if cur_obj[1] == "parse" and not cur_obj[2].startswith("增强"):
                            break
                        if cur_obj[1] == "no_sig_shape" and cur_obj[2]!="清晰":
                            break
                        if cur_obj[1] == "modality":
                            obj[-3].append(follow)
                        if cur_obj[1] == "anatomy" and len(obj[-3])!=0:
                            break
                    for follow in range(idx+1,len(new_cur_group)):
                        cur_obj = new_cur_group[follow]
                        if cur_obj[1] == "parse" and not cur_obj[2].startswith("增强"):
                            break
                        if cur_obj[1] == "no_sig_shape":
                            if cur_obj[2]!=-1 and obj[2]!='枕大池':
                                no_sig_info[self.keytrans['no_sig_shape'][cur_obj[2]]].append(self.keytrans['anatomy'][obj[2]])
                            break
                        if cur_obj[1] == "disease":
                            no_sig_info[self.keytrans['disease'][cur_obj[2]]].append(self.keytrans['anatomy'][obj[2]])
                        if cur_obj[1] != "anatomy" and cur_obj[1]!="side":
                            break
                elif obj[1] == "modality":
                    if idx>0 and new_cur_group[idx-1][1] == "signal" and isinstance(new_cur_group[idx-1][2],str) and (new_cur_group[idx-1][2].startswith("长") or new_cur_group[idx-1][2].startswith("短")):
                        new_cur_group[idx-1][-2] = 1
                        if not isinstance(obj[3],list):
                            obj[3] = [idx-1]
                        else:
                            obj[3].append(idx-1)
                        continue
                    right = False
                    for follow in range(idx+1,len(new_cur_group)): 
                        cur_obj = new_cur_group[follow] 
                        if cur_obj[1] == "parse" or cur_obj[1]=='anatomy':
                            break
                        if cur_obj[1] == "signal" and cur_obj[-2] == 1:
                            break
                        if cur_obj[1] == "signal" and isinstance(cur_obj[2],str) and (cur_obj[2].startswith("长") or cur_obj[2].startswith("短")):
                            break
                        if cur_obj[1] == "modality" and right:
                            break
                        if cur_obj[1] == "signal":
                            right = True
                            if not isinstance(obj[3],list):
                                obj[3] = [follow]
                            else:
                                obj[3].append(follow)
                            if cur_obj[-1]=="modality":
                                break
                    if not right:
                        for follow in range(idx-1,-1,-1):
                            cur_obj = new_cur_group[follow]
                            if cur_obj[1] == "parse":
                                break
                            if cur_obj[1] == "signal":
                                if not isinstance(cur_obj[2],str):
                                    break
                                if cur_obj[0]+len(cur_obj[2]) >= obj[0] or cur_obj[2].startswith("长") or cur_obj[2].startswith("短"):
                                    cur_obj[-2] = 1
                                if not isinstance(obj[3],list):
                                    obj[3] = [follow]
                                else:
                                    obj[3].append(follow)
                                break
                elif obj[1] == "signal":
                    for follow in range(idx-1,-1,-1):
                        cur_obj = new_cur_group[follow]
                        if cur_obj[1] == "parse" or cur_obj[1]=='anatomy':
                            break
                        if cur_obj[1] == "shape":
                            flag_break =False
                            if cur_obj[2] == "周围" or cur_obj[2] == "周边":
                                for index in range(cur_obj[0]+1,obj[0]):
                                    if report[index] in ["。","；","，","、"]:
                                        flag_break = True
                                        break
                            if cur_obj[2] == "中心" or cur_obj[2] == "中央":
                                for index in range(cur_obj[0]+1,obj[0]):
                                    if report[index] in ["。","；","，","、"]:
                                        flag_break = True
                                        break
                            if not flag_break:
                                obj[3] = cur_obj[2]
                            break
            
            for ele in new_cur_group:
                if ele[1] == "anatomy":
                    temp = {"anatomy":[],"side":[], "shape":[],"modality":[],"signal":[]}
                    temp["anatomy"].append(ele[2])
                    temp["side"].append(ele[3])
                    for modal_idx in ele[-3]:
                        modal = new_cur_group[modal_idx]
                        if not isinstance(modal[3],list):
                            temp["modality"].append(modal[2])
                            temp["signal"].append(-1)
                            temp["shape"].append(-1)
                        else:
                            for s in modal[3]:
                                temp["modality"].append(modal[2])
                                sig = new_cur_group[s]
                                temp["signal"].append(sig[2])
                                temp["shape"].append(sig[3])
                    res.append(temp)
        res = list(filter((lambda i:len(i["shape"])!=0 or len(i["signal"])!=0),res))
            
        return res, no_sig_info
    

    def extract_keywords(self, keypath):
        keydic = {"anatomy":[],"side":[],"modality":[],"shape":[],"deny":[],"signal":[],"special_shape":[],"disease":[],"ignore":[],"dura":[],"scalp":[],"parse":[],"no_sig_shape":[]}
        keytrans = {"anatomy":{},"side":{-1:""},"modality":{},"shape":{-1:"unspecified"},"signal":{-1:"isointensity"},"disease":{},"no_sig_shape":{}}
        columns = [["anatomy","解剖区域","解剖区域翻译"],["side","侧别","侧别翻译"],["modality","模态","模态翻译"],["shape","形态学描述","形态学描述翻译"],["deny","否定词",""],\
                ["signal","信号","信号翻译"],["special_shape","特殊形态",""],["disease","疾病描述","疾病描述翻译"],["ignore","无关词",""],["dura","硬膜出血关键词",""],["scalp","忽略关键词",""],["parse","标点",""],["no_sig_shape","非信号形态","非信号形态翻译"]]
        data = pd.read_excel(io=keypath)
        for idx in range(len(columns)):
            for idx2,i in enumerate(data[columns[idx][1]]):
                if pd.isna(i):
                    break
                keydic[columns[idx][0]].append(i)
                if len(columns[idx][2]):
                    keytrans[columns[idx][0]][i]=data[columns[idx][2]][idx2].lower() if columns[idx][0]!='modality' else data[columns[idx][2]][idx2]
        res = {}
        for i in keydic:
            res[i] = sorted(keydic[i], key=lambda k: len(k), reverse=True)
        return res,keytrans

def test(finding,impression,language='zh-CN'):
    
    f = ARD("keywords.xlsx")
    
    if language != 'zh-CN':
        finding = GoogleTranslator(source='auto', target='zh-CN').translate(finding)
        impression = GoogleTranslator(source='auto', target='zh-CN').translate(impression)
    processed_impression = f.process_disease(impression)
    processed_finding, no_sig_info = f.process_report(finding)
    processed_finding = f.postprocess_report(processed_finding, no_sig_info)
    print("FINDING")
    print(processed_finding)
    print("IMPRESSION")
    print(processed_impression)

if __name__=='__main__':

    language = "zh-CN" # if your report is in chinese, then language is zh-CN, otherwise use the name of your language type
    finding = "your findings."
    impression = "your impression"
    test(finding,impression,language)
