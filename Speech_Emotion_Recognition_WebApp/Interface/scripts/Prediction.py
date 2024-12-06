import pandas as pd
import numpy as np
import os
import librosa
import librosa.display
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, AveragePooling1D,LSTM,BatchNormalization ,Dense,  Dropout
import tensorflow as tf 
# path to the directory
RAVD = "D:\Study\Project_II\Speech_Emotion_Recognition_Using_CNN\RAVDESS\Dataset\\"
print(RAVD)
dirl_list = os.listdir(RAVD)
# print(dirl_list)
dirl_list.sort()

emotion = []
gender = []
path = []
for i in dirl_list:
    fname = os.listdir(RAVD + i)
    for f in fname:
        part = f.split('.')[0].split('-')
        emotion.append(int(part[2]))        
        
def noise(data):
    noise_amp = 0.035*np.random.uniform()*np.amax(data)
    data = data + noise_amp*np.random.normal(size=data.shape[0])
    return data
# STRETCH
def stretch(data):
    return librosa.effects.time_stretch(data,rate=0.8)
# SHIFT
def shift(data):
    shift_range = int(np.random.uniform(low=-5, high = 5)*1000)
    return np.roll(data, shift_range)
# PITCH
def pitch(data, sampling_rate):
    return librosa.effects.pitch_shift(data, sr=sampling_rate, n_steps=0.7)
# Trying different functions above
path="D:\Study\Project_II\Speech_Emotion_Recognition_Using_CNN\Speech_Emotion_Recognition_WebApp\data.wav"
data, sample_rate = librosa.load(path)

def feat_ext(data):
    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate).T, axis=0)
    return mfcc

def get_feat(path):
    data, sample_rate = librosa.load(path, duration=2.5, offset=0.6)
    # normal data
    res1 = feat_ext(data)
    result = np.array(res1)
    #data with noise
    noise_data = noise(data)
    res2 = feat_ext(noise_data)
    result = np.vstack((result, res2))
    #data with stretch and pitch
    new_data = stretch(data)
    data_stretch_pitch = pitch(new_data, sample_rate)
    res3 = feat_ext(data_stretch_pitch)
    result = np.vstack((result, res3))
    return result

def start():
    Emotions = pd.read_csv('D:\Study\Project_II\Speech_Emotion_Recognition_Using_CNN\Speech_Emotion_Recognition_WebApp\emotion.csv')
    X = Emotions.iloc[: ,:-1].values
    Y = Emotions['labels'].values

    # As this is a multiclass classification problem onehotencoding our Y
    encoder = OneHotEncoder()
    Y = encoder.fit_transform(np.array(Y).reshape(-1,1)).toarray()
    
    from keras.models import model_from_json
    json_file = open('D:\Study\Project_II\Speech_Emotion_Recognition_Using_CNN\model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("D:\Study\Project_II\Speech_Emotion_Recognition_Using_CNN\saved_models\model3.h5")
    print("Loaded model from disk")

    #livedf= pd.DataFrame(columns=['feature']) 
    # X, sample_rate = librosa.load('RAVDESS/Dataset/Actor_01/03-01-02-02-02-01-01.wav',duration=2.5,sr=22050*2,offset=0.5)
    X,Y=[],[]
    path="D:\Study\Project_II\Speech_Emotion_Recognition_Using_CNN\Speech_Emotion_Recognition_WebApp\data.wav"

    data, sample_rate = librosa.load(path, duration=2.5,sr=22050*2, offset=0.6)
    # normal data
    res1 = feat_ext(data)
    result = np.array(res1)
    #data with noise
    noise_data = noise(data)
    res2 = feat_ext(noise_data)
    result = np.vstack((result, res2))
    #data with stretch and pitch
    new_data = stretch(data)
    data_stretch_pitch = pitch(new_data, sample_rate)
    res3 = feat_ext(data_stretch_pitch)
    result = np.vstack((result, res3))
    for ele in result:
        X.append(ele)
        Y.append(emotion)
    sample_rate = np.array(sample_rate)
    mfcc = np.mean(librosa.feature.mfcc(y=data_stretch_pitch, sr=sample_rate).T, axis=0)
    featurelive = mfcc
    livedf2 = featurelive
    livedf2= pd.DataFrame(data=livedf2)
    livedf2 = livedf2.stack().to_frame().T
    twodim= np.expand_dims(livedf2, axis=2)    
    livepreds = loaded_model.predict(twodim, batch_size=32, verbose=1)
    livepreds.shape
    livepredictions = (encoder.inverse_transform((livepreds)))
    return livepredictions[0]
