# -*- coding: utf-8 -*-



import numpy as np 
import pandas as pd



from keras_preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten, Input, GlobalMaxPool2D
from keras.optimizers import Adam
import matplotlib.pyplot as plt


import seaborn as sn
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils import class_weight


img_path = "images/"


sample_submission = pd.read_csv("sample_submission.csv")
test              = pd.read_csv("test.csv")
train             = pd.read_csv("train.csv")


train["imaged_id_fileName"] = train.image_id+".jpg"
test["imaged_id_fileName"]  = test.image_id+".jpg"


train.head()


img_height = 100
img_width = 133


batch_size = 32


labels = ["healthy","multiple_diseases","rust","scab"]


train_dataGenerator = ImageDataGenerator(rescale=1./255,
    shear_range=0,
    zoom_range=(1, 1.3),
    rotation_range = 360,
    brightness_range = (0.7, 1.3),                                                   
    horizontal_flip=True,
    vertical_flip=True,
    validation_split=0.2)

train_generator = train_dataGenerator.flow_from_dataframe(
    dataframe=train,x_col='imaged_id_fileName', y_col=labels,
    directory=img_path, target_size=(img_height, img_width),
    batch_size=batch_size,class_mode='raw', subset='training') 

validation_generator = train_dataGenerator.flow_from_dataframe(
    dataframe=train,x_col='imaged_id_fileName', y_col=labels,
    directory=img_path, target_size=(img_height, img_width),
    batch_size=batch_size, class_mode='raw', subset='validation')


validation_generator2 = train_dataGenerator.flow_from_dataframe(
    dataframe=train,x_col='imaged_id_fileName', y_col=labels,
    directory=img_path, target_size=(img_height, img_width),
    batch_size=batch_size, class_mode='raw',shuffle=False,
    sort = False, subset='validation') 


train_dataGenerator_full = ImageDataGenerator(rescale=1./255,
    shear_range=0,
    zoom_range=(1, 1.3),
    rotation_range = 360,
    brightness_range = (0.7, 1.3),                                                   
    horizontal_flip=True,
    vertical_flip=True,
    validation_split=0) 

train_generator_full = train_dataGenerator_full.flow_from_dataframe(
    dataframe=train,x_col='imaged_id_fileName', y_col=labels,
    directory=img_path, target_size=(img_height, img_width),
    batch_size=batch_size, class_mode='raw', subset='training') 


test_dataGenerator = ImageDataGenerator(rescale=1./255)

test_generator = test_dataGenerator.flow_from_dataframe(
    dataframe=test,x_col='imaged_id_fileName', y_col=labels,
    directory=img_path, shuffle = False, sort = False,
    target_size=(img_height, img_width), batch_size=1, class_mode=None)


classProb =np.zeros(len(labels))
idx = 0
for k in labels:
    print(f"{k} contains {train[k].sum()} samples")
    classProb[idx] = train[k].sum()
    idx+=1


print() 
color = ['#58508d','#bc5090','#ff6361', '#ffa600'] 
plt.figure(figsize=(15,7))
plt.pie(classProb, shadow=True, explode=[0,0.5, 0, 0],labels=labels,
        autopct='%1.2f%%', colors=color, startangle=-90,
        textprops={'fontsize': 14})

class_weight_vect =np.square(1 / (classProb/classProb.sum()) )# Calculate the weight per classbased on the prior probability dervied from the training data.
class_weight_vect=class_weight_vect/np.min(class_weight_vect)


def plotImages(imgs):
    col=5
    row=2
    fig, axes = plt.subplots(row, col, figsize=(25,25))  
    axes = axes.flatten()
    for k in range(10):
        axes[k].imshow(imgs[k])
    fig.subplots_adjust(hspace=-0.75, wspace=0.2) 
    plt.show()

    

plotImageAugmentation = [validation_generator2[1][0][0] for i in range(10)] #Using validation_generator2 for consitency since shuffle is turned off.
plotImages(plotImageAugmentation)


model = Sequential()
model.add(Conv2D(35, kernel_size=(3, 3), activation='relu', kernel_initializer='glorot_uniform', 
                 bias_initializer='zeros',  input_shape=(img_height, img_width, 3), padding='same'))
model.add(Conv2D(35, (3, 3),kernel_initializer='glorot_uniform', bias_initializer='zeros', 
                 activation='relu', padding='same'))
model.add(Dropout(0.1))
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Conv2D(35, (3, 3),kernel_initializer='glorot_uniform', bias_initializer='zeros', activation='relu'))
model.add(Conv2D(35, (3, 3),kernel_initializer='glorot_uniform', bias_initializer='zeros', activation='relu'))
model.add(Dropout(0.1))
model.add(MaxPool2D(pool_size=(5, 5)))

model.add(Conv2D(50, (3, 3),kernel_initializer='glorot_uniform', bias_initializer='zeros', activation='relu'))
model.add(Conv2D(50, (3, 3),kernel_initializer='glorot_uniform', bias_initializer='zeros', activation='relu'))
model.add(Dropout(0.1))
model.add(GlobalMaxPool2D())

model.add(Dropout(0.1))
model.add(Dense(4, activation='softmax'))

optimizerAdam = Adam(lr=0.00125, amsgrad=True)

model.compile(loss="categorical_crossentropy", optimizer=optimizerAdam, metrics=["accuracy"])


model.summary()


nb_epochs = 100
history = model.fit_generator(
    train_generator,
    steps_per_epoch = train_generator.samples // batch_size,
    validation_data = validation_generator, 
    validation_steps = validation_generator.samples // batch_size,
    epochs = nb_epochs,
    class_weight=class_weight_vect)


fs = 17
fig = plt.figure(figsize=(9,5))
fig.patch.set_facecolor('xkcd:white')
plt.plot(history.history['accuracy'], color=color[0])
plt.plot(history.history['val_accuracy'], color=color[3])
plt.ylabel('Accuracy',fontsize=fs)
plt.xlabel('Epoch #',fontsize=fs)
plt.legend(['training', 'validation'],fontsize=fs)
plt.grid('both', linestyle='--')
plt.xticks(fontsize = fs)
plt.yticks(fontsize = fs)
plt.show()


fig = plt.figure(figsize=(9,5))
fig.patch.set_facecolor('xkcd:white')
plt.plot(history.history['loss'], color=color[0])
plt.plot(history.history['val_loss'], color=color[3])
plt.ylabel('Loss',fontsize=fs)
plt.xlabel('Epoch #',fontsize=fs)
plt.legend(['training', 'validation'],fontsize=fs)
plt.grid('both', linestyle='--')
plt.xticks(fontsize = fs)
plt.yticks(fontsize = fs)
plt.show()


Y_pred = model.predict(validation_generator2)
Y_pred_labels = np.argmax(Y_pred, axis=1)
y_true = np.argmax(validation_generator.labels, axis=1 )


labels_num = [0,1,2,3]
cm = confusion_matrix( y_true, Y_pred_labels, normalize='true')
sn.set(font_scale=1.4) 
sn.heatmap(cm, annot=True, annot_kws={"size": 14}, cmap="YlGnBu", xticklabels = labels, yticklabels = labels)
plt.show()


print(classification_report(y_true, Y_pred_labels))


nb_epochs = 50
history = model.fit_generator(
    train_generator_full,
    steps_per_epoch = train_generator_full.samples // batch_size,
    epochs = nb_epochs,
    class_weight=class_weight_vect)


test_predictions = model.predict_generator(test_generator)


predictions = pd.DataFrame()
predictions['image_id'] = test.image_id
predictions['healthy'] = test_predictions[:, 0]
predictions['multiple_diseases'] = test_predictions[:, 1]
predictions['rust'] = test_predictions[:, 2]
predictions['scab'] = test_predictions[:, 3]
predictions.to_csv('submission.csv', index=False)
predictions.head(10)


 from google.colab import files
 files.download("submission.csv")