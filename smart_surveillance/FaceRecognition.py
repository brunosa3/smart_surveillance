
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from itertools import chain
import json
import tensorflow as tf
import matplotlib.patches as patches
from numpy.linalg import norm
from mtcnn.mtcnn import MTCNN    
import os

# this code allocates only 1GB of your GPU to tensorflow used by face detection
# gpus = tf.config.list_physical_devices('GPU')
# if gpus:
#   # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
#     try:
#         tf.config.set_logical_device_configuration(
#             gpus[0],
#             [tf.config.LogicalDeviceConfiguration(memory_limit=1024)])
#         logical_gpus = tf.config.list_logical_devices('GPU')
#         print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
#     except RuntimeError as e:
#         # Virtual devices must be set before GPUs have been initialized
#         print(e)
  
# this code allocates only as much GPU as it needs and expands the memory when running
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:# Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

class FaceRecognition:
    def who_is_it(self, image_path, FaceNet, clf, thr=0.75, plot=False,res_model="espcn", zoom=4):

        """
        Implements face recognition for the office by finding who is the person on the image_path image.
        
        ...
        
        Parameter
        -------
        image_path : str
            path to an image
        
        database : dic 
            database containing image encodings along with the name of the person on the image as key
        
        ...
        
        Output
        -------
            min_dist : float
                the minimum distance between image_path encoding and the encodings from the database
            identity : str
                the name prediction for the person on image_path
            model_path : str
                path to the FaceNet config file path
            weight_path : str
                path to the FaceNet weight file path
        """

        # import the FaceNet model
        self.FaceNet = FaceNet
        
        # detect faces
        face = self.detect_face(image_path=image_path, res_model= res_model, zoom=zoom, plot=plot)
#         print(face.shape)

        if face is None:
            return None, ["No face visible"], None
#         plt.imshow(face)
        resolution = int(len(face)/zoom)
        print("resolution of face {}".format(resolution))
        
#         from skimage.transform import resize

#         face = resize(face, (160, 160, 3))
#         face = self.resize(face[:,:,0], 160, 160)
        
        tmp = "tmp/tmp.png"
        plt.imsave(tmp, face)
        
        # Compute the target "encoding" for the image
        encoding =  self.img_to_encoding(tmp, FaceNet)

        res = self.find_nearest([int(k) for k in clf.keys()], resolution)
        print("nearest resolution {}".format(res))
        
#         knn = KNeighborsClassifier(weights="distance", n_neighbors=5)
#         X,y = arr[res]
#         clf[res].fit(X,y)
        identity, dist, status = clf[res].predict(encoding)
        
        if status == "distance":
            print("KNN used class size to re-weight the neigbors")
        else:
            print("KNN did not use class size to re-weight the neigbors")
            
        n,w = dist[0]
        dist = list(zip(n,w))
        dist_ind = [x for x, y in enumerate(dist) if y[0] ==  identity]

        min_dist = 1/np.array(dist)[dist_ind][:,1].astype(float).mean()

        ### END CODE HERE

        if min_dist < 1/thr:
            print("Not in the database.")
            identity = ["unknown"]
            min_dist = None
        else:
            print ("it's " + ' '.join(identity) + ", the distance is " + str(min_dist))
            
        
#         proba = clf[res].predict_proba(encoding.reshape(1,-1)) #[0] this is needed for KNN model
        
#         # this needs to be uncommented when using Knn
#         thr = 1/len(proba) * 10
        

#         ind = np.where(proba>=thr)
        
#         if ind[1] != []:
#             identity = clf[res].classes_[ind[1]]
#             min_dist = proba[ind]
#         else:
#             identity = "unkwown"
#             min_dist = None

# this pice is for knn model
#         if ind != []:
#             identity = clf[res].classes_[ind]
#             min_dist = proba[ind] 
#         else:
#             identity = "unknown"
#             min_dist = None

          
        
        return min_dist, identity, encoding
    
 
    
#     def img_to_encoding(self, img): 
#         img = np.around(np.array(img) / 255.0, decimals=12)
# #         img = img.resize((160,160), Image.NEAREST)        
#         x_train = np.expand_dims(img, axis=0)
#         embedding = self.FaceNet.predict_on_batch(x_train)
#         return embedding / np.linalg.norm(embedding, ord=2)
    def img_to_encoding(self, image_path, model):
        img = tf.keras.preprocessing.image.load_img(image_path, target_size=(160, 160))
        img = np.around(np.array(img) / 255.0, decimals=12)
        x_train = np.expand_dims(img, axis=0)
        embedding = model.predict_on_batch(x_train)
        return embedding / np.linalg.norm(embedding, ord=2) 
    
    def super_resolution(self, img, model="lapsrn", zoom=8):
        import cv2
        import os
        sr = cv2.dnn_superres.DnnSuperResImpl_create()
        path = "/home/brunosa3/projects/smart_surveillance/scr/smart_surveillance/FaceRecognition/superresolution_models/"
        # print(os.listdir(path))
        path = [path + i for i in os.listdir(path) if (i.split("_")[0].lower() == model) & (int(i.split("_")[1][1]) == zoom)][0]

        sr.readModel(path) 
        sr.setModel(model, zoom) # set the model by passing the value and the upsampling ratio
        result = sr.upsample(img) # upscale the input image
        return result
    
    def detect_face(self, image_path,  res_model, zoom, plot=False):
        from PIL import Image

        if image_path.split(".")[-1].lower() == "png":
            # png have 4 channels R,G,B and alpha for transparency --> here we get rid of the aloha/transperency channel
            try:
#                 print(os.path.dirname(os.path.realpath(__file__)))
                image = Image.open(image_path).convert('RGBA')
                background = Image.new('RGBA', image.size, (255,255,255))
                alpha_composite = Image.alpha_composite(background, image)
                alpha_composite_3 = alpha_composite.convert('RGB')
                pic = np.asarray(alpha_composite_3)

            except Exception as e:
                print("[WARNING] there was an error in this image {} - maybe it is truncated?".format(image_path))
                print(e)
                return None

            
        else:
            pic = plt.imread(image_path)
            
        pic = self.super_resolution(pic, model=res_model, zoom=zoom)

        dim = pic.shape
        # create/initiate the detector, using default weights
        detector = MTCNN()
        
        # detct the face
        try:
            faces = detector.detect_faces(pic)
        except Exception as e:
            print(e)
            faces = []
            
        if len(faces) == 0:
            return None
        if len(faces) >= 1:
            
            print("[INFO] {} faces detected!!!".format(len(faces)))
            
            ind = np.argmax([faces[i]["confidence"] for i in range(len(faces))])
            faces = faces[ind]
            print(faces["box"])
            
        
         # get coordinates of detected face 
        x1, y1, width, height = faces['box']
        x2, y2 = x1 + width, y1 + height

        # recenter the detected rectangle around the face
        circles = faces['keypoints'].values() # rember keypoints contain dic of landmarks in format of tuple(x,y)
        center = (sum(k[0] for k in circles)/len(circles), sum(k[1] for k in circles)/len(circles))


        ### adjust the ratio of the rectangle towards the longer side of height and width
        ### remember input for FaceNet is 160x160 as we don t want to deform the face by resizing it
        ### we recenter the rectangle and keep same side size                
        # determine the longer side of width and height
        MAX = int(np.max([width, height]))
        x1, x2, y1, y2 = (int(center[0]-MAX/2), int(center[0]+MAX/2), int(center[1]-MAX/2), int(center[1]+MAX/2))

        # check for the cases that we go out of the image with the new rectangle

        # in case we go out of the left side of the image
        if x1 < 0: 
            # add the part which was over the left side and add it to the right side of the rectangle
            x2 += x1*-1
            # set the left point to the border of the left side of the image
            x1 = 0
        # in case we go out of the right side of the image
        if x2 > dim[1]:
            # add the part which was over the right side and add it to the left side of the rectangle
            x1 -= x2-dim[1]
            # set the right point to the border of the right side of the image
            x2 = dim[1]
        # in case we go out of the top side of the image
        if y1 < 0:    
            # add the part which was over the top part and add it to the bottom part of the rectangle
            y2 += y1*-1
            # set the top point to the boarder of the top part of the image
            y1 = 0
        # in case we go out of the bottom part of the image
        if y2 > dim[0]:
            # add the part which was over the bottom part and add it to the top part of the rectangle
            y1 -= y2-dim[0]
            # set the bottom point to the boarder of the bottom part of the image
            y2 = dim[0]
            
        if plot:
            
            import matplotlib.patches as patches
            from PIL import Image

            im = Image.open(image_path)

            # Create figure and axes
            fig, ax = plt.subplots()

            # Display the image
            ax.imshow(im)

            # Create a Rectangle patch
            rect = patches.Rectangle((x1, y1), width, height, linewidth=3, edgecolor='r', facecolor='none')

            # Add the patch to the Axes
            ax.add_patch(rect)

            plt.show()

            plt.imshow(pic[y1:y2, x1:x2])


        # crop the face out of the image 
        return pic[y1:y2, x1:x2]
    
    def resize(self, im, nR, nC):
        nR0 = len(im)     # source number of rows 
        nC0 = len(im[0])  # source number of columns 
        return [[ im[int(nR0 * r / nR)][int(nC0 * c / nC)]  for c in range(nC)] for r in range(nR)]
    
    def find_nearest(self, array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return str(array[idx])

    
def import_FaceNet(model_path='/home/brunosa3/projects/smart_surveillance/scr/smart_surveillance/FaceRecognition/model.json', 
                   weight_path = '/home/brunosa3/projects/smart_surveillance/scr/smart_surveillance/FaceRecognition/model.h5'):

    from tensorflow.keras.models import model_from_json
    json_file = open(model_path, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    FaceNet = model_from_json(loaded_model_json)
    FaceNet.load_weights(weight_path)
    return FaceNet
    #simple image scaling to (nR x nC) size
    


