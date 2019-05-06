from network import network
from PIL import Image
import numpy as np
import os
import shutil
import glob

def image_filtering(image_name):
    height, width = 128, 128
    im = np.empty(shape=(1,128,128,3))
    img = Image.open(image_name)
    img = img.resize((height, width), Image.ANTIALIAS)
    im[0,:,:,:] = np.array(img)[:,:,:3]
    
    return im

def move_image(image_path, root):
    shutil.copy(image_path, root+'static')
    os.remove(image_path)
    pass

def retrieve_image():
    root = '/app'
#    root = '/Users/emre/apps/capstone_app/'
    
    if glob.glob(root+'*.jpg'):
        image_path = glob.glob(root+'*.jpg')[0]
    elif glob.glob(root+'*.png'):
        image_path = glob.glob(root+'*.png')[0]
    elif glob.glob(root+'*.jpeg'):
        image_path = glob.glob(root+'*.jpeg')[0]
    elif glob.glob(root+'*.eps'):
        image_path = glob.glob(root+'*.eps')[0]
    
    return root, image_path
       
def emotion_predictor():
    root, image_path = retrieve_image()
    im = image_filtering(image_path)
    
    model = network()
    model.load_weights('./model_weights.h5') #load weights
    probs = list(model.predict(im)[0])
    emotion_probs = probs.copy()
    move_image(image_path, root)
    
    emotions_dict = {0:'neutral', 1:'happy', 2:'sad', 3:'surprised', 4:'fearful', 
                     5:'disgusted', 6:'angry', 7:'contempt'}
    
    emotions_dict_ = {0:'neutral', 1:'happiness', 2:'sadness', 3:'surprise', 4:'fear', 
                     5:'disgust', 6:'anger', 7:'contempt'}
    
    top1_emotion, probs[probs.index(max(probs))] = probs.index(max(probs)), 0
    top2_emotion, probs[probs.index(max(probs))] = probs.index(max(probs)), 0
    top3_emotion = probs.index(max(probs))
    
    printout1 = 'You look {} in this picture!'.format(
            emotions_dict[top1_emotion])
    printout2 = 'The other two probable emotions are {} and {}.'.format(
            emotions_dict_[top2_emotion], emotions_dict_[top3_emotion])
    
    return printout1, printout2

if __name__ == '__main__':
    result = emotion_predictor()
