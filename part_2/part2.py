# import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import os
import sys, getopt
import re
import pickle


img_height = 100 
img_width = 100
image_size = (img_height, img_width)
batch_size = 128

class_names = ['building', 'dog', 'face']

def fit(infile):

    profiles = []
    prob = dict()
    counts = {'building':{1:0,2:0,3:0},'face':{1:0,2:0,3:0},'dog':{1:0,2:0,3:0}}
    
    # train the profile model
    with open(infile) as inf:
      for count, line in enumerate(inf):
        line = line.strip()
        if not line: continue   # skip blank lines
        if count == 0: continue # skip header

        profile = dict(picType='', stars=0, predStars=0)

        info = line.split(',')
        pic = info[0]
        stars = int(info[1])

        profile['picType'] = pic
        profile['stars'] = stars

        counts[pic][stars] += 1
        profiles.append(profile)

    # calculate the probability of each star for each picure type
    for picType in counts:
        total = counts[picType][1] + counts[picType][2] + counts[picType][3]
        prob[picType] = (counts[picType][1]/total,counts[picType][2]/total,counts[picType][3]/total)

    print('\nstars pred    prob *       prob **      prob ***  picType')
    for profile in profiles:
        picType = profile['picType']

        pred = max(prob[profile['picType']])
        profile['predStars'] = prob[profile['picType']].index(pred) + 1   
 
        print('  '+str(profile['stars']) + '     ' + str(profile['predStars'])+"   {:.8f}".format(prob[profile['picType']][0])+"   {:.8f}".format(prob[profile['picType']][1]) +"   {:.8f}".format(prob[profile['picType']][2]) + '  ' + profile['picType'])

    # print results
    correct = 0
    wrong = 0
    for profile in profiles:
        if profile['stars'] == profile['predStars']:
            correct += 1
        else:
            wrong += 1

    accuracy = correct/(correct+wrong)*100

    print('\ncorrect:',correct, '\nwrong:',wrong)
    print('accuracy: ' + "{:.2f}".format(accuracy) + '%\n')

    # now train the image recognition model

    train_ds = tf.keras.utils.image_dataset_from_directory(
      'training_data/image_dataset',
      validation_split=0.2,
      subset="training",
      seed=123,
      image_size=image_size,
      batch_size=batch_size)

    val_ds = tf.keras.utils.image_dataset_from_directory(
      'training_data/image_dataset',
      validation_split=0.2,
      subset="validation",
      seed=123,
      image_size=image_size,
      batch_size=batch_size)

    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    num_classes = len(class_names)

    model = Sequential([
      layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
      layers.Conv2D(16, 3, padding='same', activation='relu'),
      layers.MaxPooling2D(),
      layers.Conv2D(32, 3, padding='same', activation='relu'),
      layers.MaxPooling2D(),
      layers.Conv2D(64, 3, padding='same', activation='relu'),
      layers.MaxPooling2D(),
      layers.Flatten(),
      layers.Dense(128, activation='relu'),
      layers.Dense(num_classes)
    ])

    model.compile(optimizer='adam',loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=['accuracy'])

    model.summary()

    epochs=30
    print('\nfitting model in ' + str(epochs) + ' epochs\n')

    history = model.fit(train_ds,validation_data=val_ds,epochs=epochs,verbose=0)

    print('\npredicting on training files\n')

    results = []
    
    # walk through the folders to get all of the training image files
    path = os.getcwd()
    for root, dirs, files in os.walk(path):
      for filename in files:
        f = os.path.join(root, filename)

        if 'jpg' not in filename: continue
        
        folders = root.split('/')
        folder = folders[-1]

        img = tf.keras.utils.load_img(f, target_size=(img_height, img_width))
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0) # Create a batch

        predictions = model.predict(img_array,verbose=0)
        score = tf.nn.softmax(predictions[0])

        results.append((filename,folder,class_names[np.argmax(score)],"{:.2f}".format(100 * np.max(score))))

        print(folder + '/' + filename,'  {:.2f}%'.format(100 * np.max(score)),' ->{}'.format(class_names[np.argmax(score)]))

    correct = 0
    wrong = 0

    for result in results:
        if result[2] in result[2]:
            correct += 1
        else:
            wrong += 1

    accuracy = correct * 100 / (correct + wrong)

    print()
    print('correct:',correct, '\nwrong:',wrong)
    print('accuracy: ' + "%.2f" % accuracy)
    print()

    # save the fit models 
    print('saving profile model')
    pickle.dump(prob,open("imageProfile.pkl","wb"))

    print('saving Keras image recognition model\n')
    model.save("image_rec.h5")   


def predictImage(infile, image_rec_model):

    # go find the image file
    path = os.getcwd()
    for root, dirs, files in os.walk(path):
        for filename in files:
            if infile == filename:
                f = os.path.join(root, filename)
                break

    img = tf.keras.utils.load_img(f, target_size=(img_height, img_width))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch

    predictions = image_rec_model.predict(img_array,verbose=0)
    score = tf.nn.softmax(predictions[0])

    folders = path.split('/')
    file = folders[-1]

    prob = np.max(score)
    class_name = class_names[np.argmax(score)]

    return prob, class_name


def pred(infile):

    profiles = []

    profileModel = pickle.load( open("imageProfile.pkl","rb"))
    image_rec_model = keras.models.load_model("image_rec.h5")

    with open(infile) as inf:
      for count, line in enumerate(inf):
        line = line.strip()
        if not line: continue   # skip blank lines
        if count == 0: continue # skip header

        profile = dict(picType='', predStars=0, programmer='', imgFile='')

        info = line.split(',')
        programmer = info[0]
        pic = info[1]

        profile['programmer'] = programmer.strip()
        profile['imgFile'] = pic.strip()

        profiles.append(profile)

    # predict stars from profile pic
    for profile in profiles:
        probI, class_name = predictImage(profile['imgFile'], image_rec_model)
        profile['picType'] = class_name 
        pred = max(profileModel[profile['picType']])
        profile['predStars'] = profileModel[profile['picType']].index(pred) + 1

    # print results
    print('Results:')
    print('prog  stars')
    for profile in profiles:
        print('  ' + profile['programmer'] + '     ' + str(profile['predStars']),'<- ' + profile['picType'])
    print()



def main(argv):
  infile = ''
  mode = ''

  try:
    opts, args = getopt.getopt(argv,"hi:m:",["inf=","mode="])
  except getopt.GetoptError:
    print('unrecognized parameter')
    print ('useage: image.py -i <inputfile> -m <mode: fit or pred>')
    print()
    sys.exit()

  for opt, arg in opts:
    if opt == '-h':
      print ('useage: image.py -i <inputfile> -m <mode: fit or pred>')
      sys.exit()
    elif opt in ("-i", "--inf"):
      infile = arg
    elif opt in ("-m", "--mode"):
      if arg not in ['fit', 'pred']:
        print('unknown mode')
        print ('useage: image.py -i <inputfile> -m <mode: fit or pred>')
        print()
        sys.exit()
      else:  
        mode = arg

  if not infile or not mode: 
    print('missing parameter')
    print ('useage: image.py -i <inputfile> -m <mode: fit or pred>')
    print()
    sys.exit()

  if mode == 'fit': fit(infile)
  elif mode == 'pred': pred(infile)


if __name__ == "__main__":
  main(sys.argv[1:])


