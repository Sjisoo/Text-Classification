import pandas as pd
import argparse

from Code.preprocessing import RawDataPreprocessing,DataPreprocessing
from Code.model import RandomForest

from pyhive import hive
import puretransport

parser = argparse.ArgumentParser()

parser.add_argument('--mode', required=True, help='train or test')
parser.add_argument('--sys', required=True, help='ess or pv')

args = parser.parse_args()

if args.mode == "train":

    resIN = []
    resLB = []

    # Preprocessing
    RDP = RawDataPreprocessing()
    
    Data_Prep = RDP.dropNA(Data_PV)
    Data_Prep = RDP.prepLabelLeft(Data_Prep)
    Data_Prep = RDP.prepLabelMid(Data_Prep)
    Data_Prep = RDP.prepLabelRight(Data_Prep)

    Data_Prep = RDP.removeInputLeft(Data_Prep)
    Data_Prep = RDP.SplitUnderbar(Data_Prep)


    DP = DataPreprocessing(mode=args.sys)

    if args.sys == 'ess':
        Data_Prep = DP.RemoveNoiseESS(Data_Prep)
        Data_Prep = DP.PrepUpper(Data_Prep)
        Data_Prep = DP.PrepNumber(Data_Prep)
    elif args.sys == 'pv':
        Data_Prep = DP.PrepPV(Data_Prep)
        Data_Prep = DP.PrepNumber(Data_Prep)

    resIN = DP.WordListToSentence(Data_Prep)
    resLB = DP.LabelToList(Data_Prep)

    real_label = set(resLB)


    # # Modeling - Random Forest
    RF = RandomForest(input=resIN,label=resLB,mode=args.sys)
    RF.Vectorizer()
    RF.TrainTestSplit()
    RF.Classifier()


elif args.mode == "test":

    
    data_input = tuple(data_infer['input'])

    resIN = []

    # Preprocessing
    Prep = Preprocessing(mode=args.sys)
    DataPrep = Prep.SplitUnderbar(data_infer)

    
    if args.sys == 'ess':
        DataPrep = Prep.RemoveNoiseESS(DataPrep)
        DataPrep = Prep.PrepUpper(DataPrep)
        DataPrep = Prep.PrepNumber(DataPrep)
    elif args.sys == 'pv':
        DataPrep = Prep.RemoveNoisePV(DataPrep)
        DataPrep = Prep.PrepPV(DataPrep)
        DataPrep = Prep.PrepNumber(DataPrep)
    else:
        print("Argument Error: (--sys) You must choose between 'ess' and 'pv'")
        exit()

    resIN = Prep.WordListToSentence(Data_Prep)


    # Modeling - Random Forest
    predictedLabels = []

    RF = RandomForest(input=resIN,mode=args.sys)
    predictedLabels = RF.Infer()

    results = {'input':data_input, 'label':predictedLabels}

    result_df = pd.DataFrame(results)
    result_df.to_csv(f"./Dataset/predicted/{args.sys.upper()}_Predicted.csv", encoding="utf-8-sig")

    print(f"{args.sys.upper()}::A predicted CSV file was generated! Please check '{args.sys.upper()}_Predicted.csv' located in 'Dataset/predicted' folder.")

else:
    print("Argument Error: (--mode) You must choose between 'train' and 'test'")
    exit()





