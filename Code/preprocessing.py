import numpy as np
import joblib


class RawDataPreprocessing:
    def __init__(self):
        """ df """

    def dropNA(self,data):
        data = data.dropna()
        data = data.reset_index()
        data = data.drop(['index'], axis=1) # axis --> 0: row, 1: column
        return data
    
    
    def prepLabelLeft(self,data):
        """ tag_kor_nm 앞 부분에 있는 [숫자] 제거 """
        idxList = data[data['label'].str.contains(']',0) == True].index.tolist()

        for i in range(len(idxList)):
            row = idxList[i]
            thisLabel = data['label'][row]
            idx = data['label'][row].find(']')

            idx = idx + 1

            if data['label'][row][idx] == " ":
                idx = idx + 1
            
            temp = thisLabel[idx:]

            data['label'][row] = temp

        return data


    def prepLabelMid(self,data):
        idxList = data[data['label'].str.contains('- ',0) == True].index.tolist()

        for i in range(len(idxList)):
            row = idxList[i]
            thisLabel = data['label'][row]

            temp = thisLabel.split('- ')[1]

            data['label'][row] = temp

        return data

    
    def prepLabelRight(self,data):
        for i in range(len(data['label'])):
            thisLabel = data['label'][i]
            tempList = thisLabel.split()
            if tempList[len(tempList)-1].isdigit() and len(tempList)>1:
                digitIdx = thisLabel.find(tempList[len(tempList)-1])

                if thisLabel[digitIdx-1] == " ":
                    digitIdx = digitIdx - 1

                temp = thisLabel[:digitIdx]

                data['label'][i] = temp

        return data


    def removeInputLeft(self,data):
        data['input'] = data.input.str.split('#').str[1]
        return data


    def SplitUnderbar(self,data):
        data['input'] = data['input'].str.split('_')

        return data




class DataPreprocessing:
    def __init__(self,mode):
        # self.Input = []
        # self.Label = []        
        self.mode = mode


    def RemoveNoiseESS(self,data):

        for i in range(len(data)):

            if data['input'][i][0] in NoiseList:
                data['input'][i].pop(0)

                if data['input'][i][0] == "POS/Global/BMS1":
                    data['input'][i].pop(0)

        return data


    def RemoveNoisePV(self,data):
        for idx in range(len(data)):
            data['input'][idx] = data['input'][idx][4]
        return data


    def PrepUpper(self,data):
        for idx in range(len(data)):
            data['input'][idx] = SplitUpper(data['input'][idx],0)
        return data
    
    
    def PrepNumber(self,data):
        for idx in range(len(data)):
            data['input'][idx] = SplitNumber(data['input'][idx],0)
        return data

    def PrepPV(self,data):
        data_len = len(data)
        
        input_prep_SF = []
        
        allTags = list(data['input'])


        allTagsLen = len(allTags)   

        # 전체 데이터 배치
        for i in range(allTagsLen):
            thisTag = allTags[i]
            thisTagLen = len(thisTag)
            FeaturesLen = len(Features)

            # Row 한 줄씩
            for j in range(thisTagLen):
                tempList = []

                thisTagElement = thisTag[j]

                while(len(thisTagElement) > 0):
                    chk = False  # for문은 돌았지만 같은 feature 발견하지 못한 경우

                    for m in range(4,0,-1):
                        if thisTagElement[:m] in Features:
                            if thisTagElement[:m] in tempList:   # temp_list에 중복 체크 (허용 안함)
                                thisTagElement = thisTagElement.replace(thisTagElement[:m],"",1) 
                            else:
                                tempList.append(thisTagElement[:m])
                                thisTagElement = thisTagElement.replace(thisTagElement[:m],"",1)
                            chk = True
                            break

                    if chk == False:
                        if len(thisTagElement) > 2:    # thisTag 길이가 2보다 큰 경우, 첫 char 처리 후 다시 실행
                            tempList.append(thisTagElement[0])
                            thisTagElement = thisTagElement.replace(thisTagElement[0],"",1)
                        else:
                            tempList.append(thisTagElement[:])
                            break
                data['input'][i] = tempList
                
        return data




    def WordListToSentence(self,data):
        input_res = []

        input = data['input']
        data_len = len(data)

        for x_idx in range(data_len):
            
            temp = ""
            
            word_list = input[x_idx]
            for w_idx in range(len(word_list)):
                temp += word_list[w_idx]
                if w_idx+1 != len(word_list):
                    temp += " "

            input_res.append(temp)
        return input_res

    
    def LabelToList(self,data):
        label_res = []

        label_to_list = []
        label_to_list = data['label'].tolist()

        params_labels_keys = list(dict.fromkeys(label_to_list))

        params_labels = {params_labels_keys[i] : i+1 for i in range(len(params_labels_keys))}
        labels_params = {i+1 : params_labels_keys[i] for i in range(len(params_labels_keys))}

        joblib.dump(labels_params,f"./Models/{self.mode}_labels_params.joblib")
    
        list_to_vec = []
        list_to_vec = [params_labels[label_to_list[i]] for i in range(len(label_to_list))]

        label_res = np.array(list_to_vec)

        # print(params_labels)
        return label_res





def SplitUpper(word_list, wl_start):
    word_list_len = len(word_list)   
    
    for wl in range(wl_start, word_list_len):
        word = word_list[wl]
        if len(word) > 2:
            for w in range(2,len(word)):
                if word[w-1].islower()==True and word[w].isupper()==True:
#                     print("START_WORD: ", word)
#                     print("raw: ", word_list)
                    word_list.pop(wl)
#                     print("pop: ", word_list)
                    word_list.insert(wl, word[:w])
#                     print("in1: ", word_list)
                    word_list.insert(wl+1, word[w:])
#                     print("in2: ", word_list)
#                     print("\n")
                    
                    return SplitUpper(word_list, wl+1)
        elif word == '-':
#             print("START_WORD: ", word_list[wl_start])
#             print("raw: ", word_list)
            word_list.pop(wl)
#             print("pop: ", word_list)
#             print("\n")
            return SplitUpper(word_list, wl)
    return word_list


# Remove Noise Number 
def SplitNumber(word_list, wl_start):
    word_list_len = len(word_list)   
    
    for wl in range(wl_start,word_list_len):
        word = word_list[wl]
        
        if word[:5]=='Alarm' or word[:2]=='LV' or word[:2]=='HV':
            return SplitNumber(word_list, wl+1)
        else:
            for w in range(len(word)):
                if ord(word[w]) >= 48 and ord(word[w]) <= 57:
                    word_list.pop(wl)
                    if w != 0:
                        word_list.insert(wl, word[:w])
                        if word[w] != word[w:]:
                            word_list.insert(wl+1, word[w+1:])
                            return SplitNumber(word_list, wl+1)
                    else:
                        if len(word) > 1:
                            word_list.insert(wl, word[w+1:])
                            return SplitNumber(word_list, wl)
                        else:
                            return SplitNumber(word_list, wl)
                          
    return word_list

# Append Noise Number
# def SplitNumber(word_list, wl_start):
#     word_list_len = len(word_list)   
    
#     for wl in range(wl_start,word_list_len):
#         word = word_list[wl]
        
#         if word[:5]=='Alarm' or word[:2]=='LV' or word[:2]=='HV':
#             return SplitNumber(word_list, wl+1)
#         else:
#             for w in range(len(word)):
#                 if ord(word[w]) >= 48 and ord(word[w]) <= 57:
# #                     print("START_WORD: ", word)
# #                     print("raw: ", word_list)
#                     word_list.pop(wl)
# #                     print("pop: ", word_list)
#                     word_list.insert(wl, word[:w])
# #                     print("in1: ", word_list)
                    
#                     if word[w] != word[w:]:
#                         word_list.insert(wl+1, word[w])
# #                         print("in2: ", word_list)
#                         word_list.insert(wl+2, word[w+1:])
# #                         print("in3: ", word_list)
#                         return SplitNumber(word_list, wl+2)

#                     word_list.insert(wl+1, word[w:])
# #                     print("in2: ", word_list)
#                     return SplitNumber(word_list, wl+2)
#     return word_list



def SplitFeatures(word_list):
    word_list = list(word_list)
    # word_list = [['STRC4'], ['L1C'], ['EGPP']]
    word_list_len = len(word_list)   

    # Features_4 = ['TAPW', 'TMPS', 'CTCH']
    # Features_3 = ['TMP', 'STR', 'ATM', 'TPF', 'FRQ']
    # Features_2 = ['L1','L2','L3','DC','HS','EG','TA','TR','HI','SI','RS','HM','DT','WN','SP','DT']
    # Features_1 = ['C','P','V','D','E','M','A']
    Features = ['TAPW', 'TMPS', 'CTCH',
            'TMP', 'STR', 'ATM', 'TPF', 'FRQ',
            'L1','L2','L3','DC','HS','EG','TA','TR','HI','SI','RS','HM','DT','WN','SP','DT',
            'C','P','V','D','E','M','A']

    word_list_len = len(word_list)   
    
    output = []
    
    for i in range(word_list_len):
        
        temp_list = []

        thisTagList = word_list[i]
        thisTagListLen = len(thisTagList)
        FeaturesLen = len(Features)

        for j in range(thisTagListLen):
            thisTag = thisTagList[j]

            while(len(thisTag) > 0):
                chk = False  # for문은 돌았지만 같은 feature 발견하지 못한 경우
                
                for m in range(4,0,-1):
                    if thisTag[:m] in Features:
                        if thisTag[:m] in temp_list:   # temp_list에 중복 체크 (허용 안함)
                            thisTag = thisTag.replace(thisTag[:m],"",1) 
                        else:
                            temp_list.append(thisTag[:m])
                            thisTag = thisTag.replace(thisTag[:m],"",1)
                        chk = True
                        break
                    
                if chk == False:
                    if len(thisTag) > 2:    # thisTag 길이가 2보다 큰 경우, 첫 char 처리 후 다시 실행
                        temp_list.append(thisTag[0])
                        thisTag = thisTag.replace(thisTag[0],"",1)
                    else:
                        temp_list.append(thisTag[:])
                        break
            output.append(temp_list)
    return output