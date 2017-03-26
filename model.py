import numpy as np
import pandas as pd
from random import randint
import random
import PIL
from PIL import ImageOps
from PIL import ImageEnhance
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Convolution2D, ELU, Flatten, Dropout, Dense, Lambda, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img


#constants and globals
BATCH_SIZE = 32
EPOCHS = 20
NUM_BATCHES_PER_EPOCH = 512

#Built in Keras iterator function to yield dynamically generated images. 
datagen = ImageDataGenerator(
    rotation_range=5,
    width_shift_range=0.1,
    height_shift_range=0.1,
        #shear_range=0.2,
        #zoom_range=0.2,
    channel_shift_range = 0.2 ,
    horizontal_flip=False,
    fill_mode='constant',
    cval=0.0)

#subfunction to return image and steering angle to the data generator 
def grab_preprocess_raw_image(data):
    global datagen
    angle = data['steering'].iloc[0]

    #randomly select left,right or center methods
    position = np.random.choice(['left', 'right','center'])
    if position =='left':
        angle += .25 
    elif position == 'right':
        angle -= .25 
    
    #Randomly transform image to a dynamically adjusted one. 
    img = load_img(data[position].iloc[0].strip())
    if np.random.uniform() < .5 :
        x = img_to_array(img)  
        x = x.reshape((1,) + x.shape) 
        imgy = datagen.flow(x, batch_size=1).next()
        img = array_to_img(imgy[0])
    
    #crop and resize    
    w, h = img.size
    img = img.crop((0, h*0.40, w, h-25))
    img = img.resize((32, 16), PIL.Image.ANTIALIAS)
   
    #randomly flip the image and angle horizontally
    if np.random.uniform() < .5 :
        img = ImageOps.mirror(img)
        angle = -angle

        
    #30% chance to enhance the brightness
    if np.random.uniform() < .3:
        enhancer = ImageEnhance.Brightness(img)
        if np.random.uniform() < .5:
            img = enhancer.enhance(0.7)
        else:
            img = enhancer.enhance(1.5)

    #reduce the bits per channel for more efficient computations
    img = ImageOps.posterize(img, 4)  
    return img, angle

#The data generator that is provided to the trainer
#First it creates a batch size array matrix in itialized to zeros as well as a label array
#Then it populates these with the image generator function above
#There is a 10% chance the data will be pulled from the entire set, the other 90% chance is 
#pull from either a left or right direction turns with 50% chance of pulling either direction
def image_angle_gen(data, bsize):
    while(1):
        X = np.zeros((bsize, 16, 32, 3), dtype = np.float32)
        y = np.zeros((bsize), dtype = np.float32)
        for i in range(bsize):
            if np.random.uniform() < .1 :  
                    row = data.sample(n=1)
            else:
                if np.random.uniform() < .5:
                    row = data[(data['steering'] > 0.05) ].sample(n=1)
                else:
                    row = data[(data['steering'] < -0.05 )].sample(n=1)
            X[i],y[i]  = grab_preprocess_raw_image(row)
        yield X,y

#This is the Keras model definition. Consists of a 
#Normalization lambda layer
#convolution layer with a 2x2 pool
#anothr convo layer with constant pool
#a dropout of 20% 
#another convo layer with constant pool
#flatten to long array
#a fully connected layer followed by another dropout of 30%
#following by 2 more fully connected layer, ending in a single length output (our prediction)

#the model is then compiled with an Adam optimizer and a loss function of mean square error
def get_model():
    model = Sequential()
    model.add(Lambda(lambda x: x/127.5 - 1., input_shape=(16, 32, 3)))
    model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same", activation='elu'))
    model.add(Convolution2D(16, 3, 3, subsample=(1, 1), border_mode="same", activation='elu'))
    model.add(Dropout(.2))
    model.add(Convolution2D(16, 3, 3, subsample=(1, 1), border_mode="same", activation='elu'))
    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Dropout(.3))
    model.add(ELU())
    model.add(Dense(512, activation='elu'))
    model.add(Dense(1))

    opt = Adam(lr=0.001)
    model.compile(optimizer=opt, loss='mse', metrics=[])

    return model
#Our function to train the model and save the results. 
#First the data is shuffles to rid any ordering biases
#We then split the data into a train and validation set. We didnt use a test set here since our
#testing is performed manually on the track
#We create 2 instances of the data generator above, one for the training data and one for the validation datagen
#We get an instance of our model defnition above then we use the model's function 'fit_generator' since we are 
#supplying a data generator. 

#we save the model after training
def run_trainer(df):
    
    df = df.sample(frac=1).reset_index(drop=True)
    Xysplit = 0.8
    rows = int(df.shape[0]*Xysplit)

    X_train = df.loc[0:rows-1]
    X_val = df.loc[rows:]

    train_gen = image_angle_gen(X_train, bsize=BATCH_SIZE)
    val_gen = image_angle_gen(X_val, bsize=BATCH_SIZE)
   
    model = get_model()

    samples = BATCH_SIZE * NUM_BATCHES_PER_EPOCH
    model.fit_generator(train_gen, validation_data=val_gen,samples_per_epoch=samples, nb_epoch=EPOCHS, nb_val_samples=1000)

    model.save_weights('model.h5')  
    with open('model.json', 'w') as outfile:
        outfile.write(model.to_json())

df = pd.read_csv('driving_log.csv')
run_trainer(df)