from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib


TEST_SIZE = 0.2
RANDOM_SEED = 42

class RandomForest:
    def __init__(self,input,mode,label=[]):
        self.input = input
        self.label = label
        self.inputFeatures = []

        self.mode = mode.upper()

        self.train_input = []
        self.train_label = []

        self.eval_input = []
        self.eval_label = []

    def Vectorizer(self):
        vectorizer = CountVectorizer(analyzer="word", max_features=10000)
        self.inputFeatures = vectorizer.fit_transform(self.input)
        joblib.dump(vectorizer,f"./Models/{self.mode}_RandomForest_Vectorizer.joblib")


    def TrainTestSplit(self):
        self.train_input, self.eval_input, self.train_label, self.eval_label = train_test_split(self.inputFeatures, self.label, test_size=TEST_SIZE, random_state=RANDOM_SEED)

    def Classifier(self):
        classifier = RandomForestClassifier(n_estimators=100)
        classifier.fit(self.train_input, self.train_label)
        joblib.dump(classifier,f"./Models/{self.mode}_RandomForest_Classifier.joblib")
        print("Accuracy: %f" % classifier.score(self.eval_input,self.eval_label))

    
    def Infer(self):
        vectorizer = joblib.load(f"./Models/{self.mode}_RandomForest_Vectorizer.joblib")
        self.inputFeatures = vectorizer.transform(self.input)

        classifier = joblib.load(f"./Models/{self.mode}_RandomForest_Classifier.joblib")
        result = classifier.predict(self.inputFeatures)
        
        labels_params = joblib.load(f"./Models/{self.mode}_labels_params.joblib")
        
        result_to_label = []

        
        for i in range(len(result)):
            result_to_label.append(labels_params[int(result[i])])
            # print("< Infer", i+1, " >")
            # print("Tag   : ", self.input[i])    
            # print("Param : ",labels_params[int(result[i])], "\n")

        return result_to_label
