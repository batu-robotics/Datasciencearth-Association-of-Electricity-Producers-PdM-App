# PdM Analyses Software
# Designed by: SUMERLabs

#%% Importing Req'd Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split,cross_val_score,RepeatedKFold
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

#%% Constructing Data Analyses Class
class Data:
    
    # Initializer
    def __init__(self,filename1,filename2,percent):
        self.filename1=filename1
        self.filename2=filename2
        self.percent=percent
        self.le=LabelEncoder()
        self.sc=StandardScaler()
        self.cv=RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
        
    # Opening Data
    def open_file(self):
        self.train_dataframe=pd.read_csv(self.filename1)
        self.test_dataframe=pd.read_csv(self.filename2)
        self.columns=self.train_dataframe.columns
    
    # Label Encoding
    def label_encode(self):
        self.le_list1=self.le.fit(self.train_dataframe.iloc[:,-8].values)
        self.le_list1=self.le.transform(self.train_dataframe.iloc[:,-8].values)
        
        self.le_list2=self.le.fit(self.train_dataframe.iloc[:,-4].values)
        self.le_list2=self.le.transform(self.train_dataframe.iloc[:,-4].values)
        
        
        self.new_dataframe_train=np.column_stack((self.train_dataframe.iloc[:,:-8].values,
                                            self.le_list1,
                                            self.train_dataframe.iloc[:,-7:-4].values,
                                            self.le_list2,
                                            self.train_dataframe.iloc[:,-3:].values
                                            ))

        self.new_dataframe_train=pd.DataFrame(self.new_dataframe_train[:,2:],columns=self.columns[2:]).astype(float)
        

        self.le_list3=self.le.fit(self.test_dataframe.iloc[:,-8].values)
        self.le_list3=self.le.transform(self.test_dataframe.iloc[:,-8].values)
        
        self.le_list4=self.le.fit(self.test_dataframe.iloc[:,-4].values)
        self.le_list4=self.le.transform(self.test_dataframe.iloc[:,-4].values)
        
        
        self.new_dataframe_test=np.column_stack((self.test_dataframe.iloc[:,:-8].values,
                                            self.le_list3,
                                            self.test_dataframe.iloc[:,-7:-4].values,
                                            self.le_list4,
                                            self.test_dataframe.iloc[:,-3:].values
                                            ))
        
        self.new_dataframe_test=pd.DataFrame(self.new_dataframe_test[:,2:],columns=self.columns[2:]).astype(float)

    # Analysing Data
    def analyse_data(self):
        self.new_dataframe_train=self.new_dataframe_train.dropna()
        self.new_dataframe_test=self.new_dataframe_test.dropna()

        self.dependent_train=self.new_dataframe_train.iloc[:,-4:-2]        
        self.independent_train=self.new_dataframe_train.drop(self.dependent_train, axis=1)

        self.dependent_test=self.new_dataframe_test.iloc[:,-4:-2]        
        self.independent_test=self.new_dataframe_test.drop(self.dependent_test, axis=1)        
        
        sns.heatmap(self.independent_train.corr(),annot=False)
        
        plt.figure("Histogram of Classes")
        plt.title("Distro of Error Modes")
        sns.histplot(self.dependent_train.iloc[:,-2].values)
        plt.grid(True)
    
    # Feature Selection based on Correlation Matrix
    def features(self):
        self.correlation_matrix=self.independent_train.corr().abs()
        self.upper_triangle=self.correlation_matrix.where(np.triu(np.ones(self.correlation_matrix.shape),k=1).astype(np.bool))
        
        self.to_drop_features=[column for column in self.upper_triangle.columns if any(self.upper_triangle[column] > 0.75)]
        print("\n Auto-Drop Features \n",self.to_drop_features,"\n")
        
        self.filtered_train=self.independent_train.drop(self.to_drop_features,axis=1)
        self.filtered_test=self.independent_test.drop(self.to_drop_features,axis=1)
            
    # Training Models and Test Results
    def cv_models(self):
        self.model1=SVC()
        self.model2=DecisionTreeClassifier()
        self.model3=RandomForestClassifier()
        self.model4=XGBClassifier()
        
        self.scores1=cross_val_score(self.model1,self.filtered_train,self.dependent_train.iloc[:,0],scoring='accuracy', cv=self.cv, n_jobs=-1)
        self.scores2=cross_val_score(self.model2,self.filtered_train,self.dependent_train.iloc[:,0],scoring='accuracy', cv=self.cv, n_jobs=-1)
        self.scores3=cross_val_score(self.model3,self.filtered_train,self.dependent_train.iloc[:,0],scoring='accuracy', cv=self.cv, n_jobs=-1)
        self.scores4=cross_val_score(self.model4,self.filtered_train,self.dependent_train.iloc[:,0],scoring='accuracy', cv=self.cv, n_jobs=-1)
        
        self.all_scores=np.column_stack((self.scores1,self.scores2,self.scores3,self.scores4))
        self.all_scores=pd.DataFrame(self.all_scores,columns=['SVC','DTC','RFC','XGBC'])
        
        plt.figure('Cross Validation Results')
        plt.plot(self.all_scores)
        plt.grid(True)
        plt.legend(self.all_scores.columns)
        
    # Batch Analyses for Test Values
    def batch_analyses(self):
        self.x_train,self.x_test,self.y_train,self.y_test=train_test_split(self.filtered_train,self.dependent_train.iloc[:,0],test_size=self.percent,shuffle=True,random_state=None)

        self.x_train=self.sc.fit_transform(self.x_train)
        self.x_test=self.sc.transform(self.x_test)
        
        self.model1.fit(self.x_train,self.y_train)
        self.model2.fit(self.x_train,self.y_train)
        self.model3.fit(self.x_train,self.y_train)
        self.model4.fit(self.x_train,self.y_train)
        
        self.svc=self.model1.predict(self.x_test)
        self.dtc=self.model2.predict(self.x_test)
        self.rfrc=self.model3.predict(self.x_test)
        self.xgbc=self.model4.predict(self.x_test)
        
        self.confusion_matrix_svc_1=confusion_matrix(self.y_test,self.svc)
        self.confusion_matrix_dtc_1=confusion_matrix(self.y_test,self.dtc)
        self.confusion_matrix_rfrc_1=confusion_matrix(self.y_test,self.rfrc)
        self.confusion_matrix_xgbc_1=confusion_matrix(self.y_test,self.xgbc)
        
        print('Validation Data Results')
        print("\n SVC Results \n",self.confusion_matrix_svc_1,"\n")
        print("\n DTC Results \n",self.confusion_matrix_dtc_1,"\n")
        print("\n RFC Results \n",self.confusion_matrix_rfrc_1,"\n")
        print("\n XGBC Results \n",self.confusion_matrix_xgbc_1,"\n")

        # for the test dataframe
        self.svc_2=self.model1.predict(self.filtered_test)
        self.dtc_2=self.model2.predict(self.filtered_test)
        self.rfrc_2=self.model3.predict(self.filtered_test)
        self.xgbc_2=self.model4.predict(self.filtered_test)
        
        self.confusion_matrix_svc_2=confusion_matrix(self.dependent_test.iloc[:,0],self.svc_2)
        self.confusion_matrix_dtc_2=confusion_matrix(self.dependent_test.iloc[:,0],self.dtc_2)
        self.confusion_matrix_rfrc_2=confusion_matrix(self.dependent_test.iloc[:,0],self.rfrc_2)
        self.confusion_matrix_xgbc_2=confusion_matrix(self.dependent_test.iloc[:,0],self.xgbc_2)
        
        print('Test Dataframe Results')
        print("\n SVC Results \n",self.confusion_matrix_svc_2,"\n")
        print("\n DTC Results \n",self.confusion_matrix_dtc_2,"\n")
        print("\n RFC Results \n",self.confusion_matrix_rfrc_2,"\n")
        print("\n XGBC Results \n",self.confusion_matrix_xgbc_2,"\n")
        
        self.all_scores_test=np.column_stack((self.dependent_test.iloc[:,0].values,self.svc_2,self.dtc_2,self.rfrc_2,self.xgbc_2))
        self.all_scores_test=pd.DataFrame(self.all_scores_test,columns=['Test Data','SVC','DTC','RFC','XGBC'])
      
    # Main Class
    def main_program(self):
        self.open_file()
        self.label_encode()
        self.analyse_data()
        self.features()
        self.cv_models()
        self.batch_analyses()
        
        return self.train_dataframe,self.test_dataframe,\
            self.columns,self.new_dataframe_train,\
            self.new_dataframe_test,self.independent_train,\
            self.dependent_train,self.independent_test,\
            self.dependent_test,\
            self.filtered_train,self.filtered_test,\
            self.all_scores,self.all_scores_test
        
# Main()
file1="ALLtrainMescla5D-1.csv"
file2="ALLtestMescla5D.csv"
percent=0.2
d=Data(file1,file2,percent)
train,test,head,new_train,new_test,ind_train,dep_train,ind_test,dep_test,filt_train,filt_test,scores1,scores_all=d.main_program()
    