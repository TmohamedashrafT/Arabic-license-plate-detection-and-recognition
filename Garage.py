
import cv2
from tensorflow.keras.applications import MobileNetV2
import torch
import tensorflow as tf
from keras import layers
import numpy as np

plate_detection_model = torch.hub.load('yolov5', 'custom', path='D:/GradProject/plate detection weights/best.pt', source='local') 
Segment_model = torch.hub.load('yolov5', 'custom', path='D:/GradProject/sympol_segmentation weights/best.pt', source='local')
input_shape=(32, 32, 3)
### numbers_model
MobileNetV2_numbers=MobileNetV2(include_top=False,input_shape=input_shape)
inputs = tf.keras.Input(shape=input_shape)
x_number = MobileNetV2_numbers(inputs)
x_number = layers.Flatten()(x_number)
x_number = layers.Dense(1024, activation='relu')(x_number)
x_number = layers.Dense(1024, activation='relu')(x_number)
x_number = layers.Dense(1024, activation='relu')(x_number)
x_number = layers.Dropout(0.3)(x_number)
outputs = layers.Dense(9, activation='softmax')(x_number)
numbers_model = tf.keras.Model(inputs, outputs)
numbers_model.load_weights("D:/GradProject/numbers weights/mobile_numbers.h5")

####  letters models
MobileNetV2_letter= MobileNetV2(include_top=False,input_shape=input_shape)
inputs_letter     = tf.keras.Input(shape=input_shape)
x_letters         = MobileNetV2_letter(inputs_letter)
x_letters         = layers.Flatten()(x_letters)
x_letters         = layers.Dense(1024, activation='relu')(x_letters)
x_letters         = layers.Dense(1024, activation='relu')(x_letters)
x_letters         = layers.Dense(1024, activation='relu')(x_letters)
x_letters         = layers.Dropout(0.3)(x_letters)
outputs_letters   = layers.Dense(17, activation='softmax')(x_letters)
letters_model     = tf.keras.Model(inputs_letter, outputs_letters)
letters_model.load_weights("D:/GradProject/letter weights/model_letters_mobile.h5")
letters={0:'A',1:'B',2:'G',3:'D',4:'R',5:'S',6:'C',7:'T',8:'E',9:'F',10:'K',11:'L',12:'M',13:'N',14:'H',15:'W',16:'Y'}
def plate_number(img):
    result=plate_detection_model(img)
    x   = result.xyxy
    result.crop()
    img_path = str(result.path)
    print(img_path)
    try:
        img_res = cv2.imread(img_path)
        #img_res = cv2.cvtColor(img_res, cv2.COLOR_BGR2RGB)
        result = Segment_model(img_path)
        h,w,_= img_res.shape
        prediction=''
        pred=''
        result=torch.sort(result.xyxy[0],0)
        for i in range(len(result[0])):
            t=result[0][i].type(torch.cuda.IntTensor)
            let = img_res[t[1]:t[3] ,t[0]:t[2]]
            img_cropped=cv2.resize(let,(32,32))
            img_cropped=np.array(img_cropped).reshape(-1,32,32,3)/255.0
            if t[2]<=int(w/2)or (len(result[0])==7 and i<4):
                    number=numbers_model.predict(img_cropped)
                    number= np.argmax(number, axis=-1)
                    prediction=prediction+str(number[0]+1)
            else:
                    letter=letters_model.predict(img_cropped)
                    letter=np.argmax(letter,axis=-1)
                    pred =pred+str(letters[letter[0]])

        act = (prediction+pred)
        w, h = cv2.getTextSize(act, 0, fontScale=3, thickness=2)[0]
        x1,y1,x2,y2=int(x[0][0][0]),int(x[0][0][1]),int(x[0][0][2]),int(x[0][0][3])
        cv2.rectangle(img, (x1+2,y1),(x2+2, y1-int((y2-y1)/4)), (0,0,255) ,-1,cv2.LINE_AA)  # filled
        cv2.rectangle(img, (int(x[0][0][0]),int(x[0][0][1])),(int(x[0][0][2]), int(x[0][0][3])), (0,0,255),    thickness=3)  # filled
        cv2.putText(img, act, (x1-2, y1), cv2.FONT_HERSHEY_PLAIN, 3, (0,255,0), 2)
    except:
        return img
    torch.cuda.empty_cache()
    return img