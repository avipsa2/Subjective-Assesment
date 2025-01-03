import math
import pandas as pd
import tensorflow as tf
import numpy as np
import pickle
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
import sklearn.metrics
import warnings
from tensorflow.keras.losses import MeanSquaredLogarithmicError
from tensorflow.keras import initializers

def generate_DNNmodel():
    
    # Reading the final scores from the all three pipelines/modules
    df = pd.read_csv('../results/FinalScoringDataset.csv')

    # scoring relaxed, constrained to one decimal place 
    def roundVal(X):
        y = X.to_list()
        for i in range(0, len(y)):
            y[i] = round(y[i], 1)
        return y
    
     
    keyword = ['keybert_score', 'yake_score']
    semantic = ['sbert_score', 'simcse_score', 'hf_score']
    NLP = ['nltk_score', 'spacy_score', 'cam_score']
    
    data = []
    
    for i in range(0, len(keyword)):
        for j in range(0, len(semantic)):
            for k in range(0, len(NLP)):
                arr = [keyword[i], semantic[j], NLP[k]]
                df1 = df[[keyword[i], semantic[j], NLP[k], 'Actual Score (0-10)']]
                df1[keyword[i]] = roundVal(df1[keyword[i]])
                df1[semantic[j]] = roundVal(df1[semantic[j]])
                df1[NLP[k]] = roundVal(df1[NLP[k]])
                warnings.simplefilter(action='ignore')
                X = df1[[keyword[i], semantic[j], NLP[k]]]
                y = df1['Actual Score (0-10)']
      
                # using the train test split function
                X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=9,
                              train_size=0.8, shuffle=True)
    
                model = Sequential()
                model.add(Dense(units=8, input_dim=3, kernel_initializer='normal', activation='relu'))
                model.add(Dense(units=16, kernel_initializer=initializers.RandomNormal(stddev=1), bias_initializer=initializers.Zeros(), activation='tanh'))
                model.add(Dense(units=32, kernel_initializer='normal', activation='relu'))
                model.add(Dense(units=64, kernel_initializer=initializers.RandomNormal(stddev=1), bias_initializer=initializers.Zeros(), activation='tanh'))
                model.add(Dense(1, kernel_initializer='normal'))
                model.compile(loss='mean_squared_error', optimizer='adam')
                model.fit(X_train, y_train ,batch_size = 6, epochs = 100, verbose=0)
                if(i==1 and j==1 and k==2): 
                    model.summary()
                    # save the model to disk
                    trainedMLmodelFilename = '../results/SubjectiveAnswerChecker.sav'
                    pickle.dump(model, open(trainedMLmodelFilename, 'wb'))
                    print("Model generated Successfully!!!")
                rmseTest=math.sqrt(sklearn.metrics.mean_squared_error(y_test/10, np.round(model.predict(X_test),1)/10))/5 
                rmseTrain=math.sqrt(sklearn.metrics.mean_squared_error(y_train/10, np.round(model.predict(X_train),1)/10))/5  
                rmseTotal=math.sqrt(sklearn.metrics.mean_squared_error(y/10, np.round(model.predict(X),1)/10))/5  
                arr.append(rmseTrain)
                arr.append(rmseTest)
                arr.append(rmseTotal)
                data.append(arr)
                tf.keras.backend.clear_session()
                
    dataframe  = pd.DataFrame(data, columns=['keyword', 'similarity', 'NLP', 'testingError/ANN', 'trainingError/ANN', 'totalError/ANN'])
    print(dataframe)

    # Saving the results in form of a csv file to be in neural network
    dataframe.to_csv('../results/ScoresErrorChart.csv', index=False)
    