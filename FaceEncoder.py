
class GenerateFaces: 
        
    # The init method or constructor
    def __init__(self, file_path , output="/mnt/golem/frodo/cropped_faces/", thr=0.95, plot=True):
        import os
        import json

        if os.path.isfile("/mnt/golem/frodo/Database/FaceDB.json"):
            self.DB = json.loads(open("/mnt/golem/frodo/Database/FaceDB.json").read())
            self.n = len(self.DB) 
        else:
            self.DB = None
            self.n = 0
        
        self.path = file_path
        self.output_dir = output
        self.thr = thr
        self.plot = plot
        self.container = {}
        self.import_FaceNet()
        self.batch = 0
#         self.DB = self.get_existingDB(self, path=None)
        
    def run(self, archive = True, batch_size = 100):
        from tqdm import tqdm 
        import matplotlib.pyplot as plt
        from mtcnn.mtcnn import MTCNN
        from PIL import Image
        from pillow_heif import register_heif_opener
        import numpy as np

                        
        for pics in tqdm(self.path):
            
            if pics in list(set([v["path_pic_orig"] for v in self.DB.values()])):
                print("this image is alredy in the Database {}".format(pics))
                continue
            
            # load the photograph
            self.filename = pics
#             print(self.filename)

            # load image from file
            if ("HEIC" == self.filename.split(".")[-1]) | ("heic"== self.filename.split(".")[-1]):

                register_heif_opener()

                try:
                    image = Image.open(self.filename)
                    self.pixels = image.__array__()                
                except Exception as e:
                    print("there was an error in this image {} - maybe it is truncated?".format(self.filename))
                    print(e)
                    continue
                
                
            elif ("PNG" == self.filename.split(".")[-1]) | ("png"== self.filename.split(".")[-1]):
                
                # png have 4 channels R,G,B and alpha for transparency --> here we get rid of it
                try:
                    image = Image.open(self.filename).convert('RGBA')
                    background = Image.new('RGBA', image.size, (255,255,255))
                    alpha_composite = Image.alpha_composite(background, image)
                    alpha_composite_3 = alpha_composite.convert('RGB')
                    
                except Exception as e:
                    print("there was an error in this image {} - maybe it is truncated?".format(self.filename))
                    print(e)
                    continue
                
                
                self.pixels = np.asarray(alpha_composite_3)

            else:
                try:
                    self.pixels = plt.imread(self.filename)
                except Exception as e:
                    print("there was an error in this image {} - maybe it is truncated?".format(self.filename))
                    print(e)
                    continue
                
                
            # create the detector, using default weights
            detector = MTCNN()
            # detect faces in the image
            self.faces = detector.detect_faces(self.pixels)
            # display faces on the original image
            self.crop_faces_centered_boxes_for_encoding()
            
            
            if archive: 
                if self.batch > batch_size:
                    print("archive batch")
                    self.archive_database()
                    self.batch = 0
                else:
                    self.batch += 1
            
#         self.get_database()
#         self.get_existingDB()
            
        if archive:
            print("preparing final archive of FaceDB ... \nlooped through {} pictures, detected {} faces in {} pictures".format(len(self.path), len(self.container), len(np.unique([v["path_pic_orig"] for v in self.container.values()]))))
            self.archive_database()
            print("archived 'FaceDB' sucessfully!!!")
            print("There are currently {} faces in FaceDB encoded".format(len(self.container)))
            
        else:
            print("DB was not archived")
        
    # draw each face separately
    def crop_faces_centered_boxes_for_encoding(self):
        import cv2
        import matplotlib.pyplot as plt
#         import uuid
        import numpy as np
        import os
        
        # input dimension
        dim = self.pixels.shape

        for i in range(len(self.faces)):
            
            # if confident in face detection
            if self.faces[i]['confidence'] > self.thr:
                
                # generate unique id for detected face
#                 ID = uuid.uuid4().int
                self.n += 1
                ID = self.n
                
#                 while True:
#                     if ID in self.container.keys():
#                         ID = uuid.uuid4().int
#                     else:
#                         break
                    
                
                # get coordinates only
                x1, y1, width, height = self.faces[i]['box']
                x2, y2 = x1 + width, y1 + height

                # recenter the detected rectangle around the face
                circles = self.faces[i]['keypoints'].values()
                center = (sum(k[0] for k in circles)/len(circles), sum(k[1] for k in circles)/len(circles))

                # adjust the ratio of the rectangle towards the onger side of height and width
                # remember input for FaceNet is 160x160 as we don t want to deform the face by resizing it
                # we recenter the rectangle and keep same side size
                MAX = int(np.max([width, height]))
                x1, x2, y1, y2 = (int(center[0]-MAX/2), int(center[0]+MAX/2), int(center[1]-MAX/2), int(center[1]+MAX/2))

                # check for the cases that we go out of limit with the rectangle
                if x1 < 0:                
                    x2 += x1*-1
                    x1 = 0

                if x2 > dim[1]:
                    x1 -= x2-dim[1]
                    x2 = dim[1]

                if y1 < 0:                
                    y2 += y1*-1
                    y1 = 0

                if y2 > dim[1]:
                    y1 -= y2-dim[0]
                    y2 = dim[0]


                # crop the face out of the image 
                crop = self.pixels[y1:y2, x1:x2]

                output_dir = self.output_dir + str(ID) + "_cropped_face.png"
                cv2.imwrite(output_dir, cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
                
                self.encodings = self.img_to_encoding(output_dir)
                
                if self.DB != None:
                    match = [k for k,v in self.DB.items() if v["encodings"] == self.encodings.tolist()]
                    if len(match) > 0:
                        print("This face alredy exist in the database {}".format(match[0]))
                        os.remove(output_dir)
                    else:
                        self.container[ID] = {"path_pic_orig": self.filename,
                                              "path_croped_pic": output_dir,
                                     "encodings": self.encodings.tolist(),
                                     "recentered_rectangle": [x1, x2, y1, y2],
                                     "detector": self.faces[i]}
                
                else:                
                    self.container[ID] = {"path_pic_orig": self.filename,
                                          "path_croped_pic": output_dir,
                                     "encodings": self.encodings.tolist(),
                                     "recentered_rectangle": [x1, x2, y1, y2],
                                     "detector": self.faces[i]}
                
                if self.plot:
                    # define subplot
                    plt.subplot(1, len(self.faces), i+1)
                    plt.axis('off')
                    # plot face
                    plt.imshow(crop)
                
            else:
                x1, y1, width, height = self.faces[i]['box']
                x2, y2 = x1 + width, y1 + height
                output_dir = self.output_dir + "less_confident_faces/" + str(self.faces[i]['confidence']) + "_cropped_face.png"
                cv2.imwrite(output_dir, cv2.cvtColor(self.pixels[y1:y2, x1:x2], cv2.COLOR_RGB2BGR))
                
                print("This face did not had high confidence {}".format(self.faces[i]['confidence']))
                
                
    def import_FaceNet(self, model_path='../dat/models/keras-facenet-h5/model.json', 
                       weight_path = '../dat/models/keras-facenet-h5/model.h5'):

        from tensorflow.keras.models import model_from_json
        json_file = open(model_path, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.FaceNet = model_from_json(loaded_model_json)
        self.FaceNet.load_weights(weight_path)

    #tf.keras.backend.set_image_data_format('channels_last')
    def img_to_encoding(self, image_path):
        import tensorflow as tf  
        import numpy as np
        img = tf.keras.preprocessing.image.load_img(image_path, target_size=(160, 160))
        img = np.around(np.array(img) / 255.0, decimals=12)
        x_train = np.expand_dims(img, axis=0)
        embedding = self.FaceNet.predict_on_batch(x_train)
        return embedding / np.linalg.norm(embedding, ord=2)
    
#     def combine_dicts(self, *dicts):
#         from functools import reduce
#         return reduce(lambda dict1, dict2: dict(zip(dict1.keys() + dict2.keys(), dict1.values() + dict2.values())), dicts)

    
    def archive_database(self):
        import json
        DB = self.get_database()        
        with open('/mnt/golem/frodo/Database/FaceDB.json', 'w') as fp:
            json.dump(DB, fp)
            
    def get_database(self):
        if self.DB != None:
            self.container.update(self.DB)            
        return self.container

                


######################################################################################################
###
###
######################################################################################################

class FaceCluster:
     
    def __init__(self, file_path="/mnt/golem/frodo/Database/FaceDB.json", IDs = None):
        import json
        import os
               
        if not (os.path.isfile(file_path) and os.access(file_path, os.R_OK)):
            print('The input encoding file, ' + str(path) + ' does not exists or unreadable')
            exit()
            
        self.DB = json.loads(open(file_path).read())
        if IDs != None:
            self.DB = {k: v for k,v in self.DB.items() if k in IDs}
        
        self.encodings = np.vstack([np.array(v["encodings"]) for k,v in self.DB.items()]) 
     
    def Cluster(self):
        from sklearn.cluster import DBSCAN
        import hdbscan
 
        NumberOfParallelJobs = -1
 
        # cluster the embeddings
        print("[INFO] Clustering")
        clt = hdbscan.HDBSCAN(min_cluster_size=10)
#         clt = DBSCAN(eps=0.75, min_samples=10,
#                       n_jobs = NumberOfParallelJobs)
                       
#         clt.fit(self.encodings)

        (labelIDs, labelIDs_freq) = np.unique(clt.fit_predict(self.encodings), return_counts=True)
        # determine the total number of
        # unique faces found in the dataset
#         (labelIDs, labelIDs_freq) = np.unique(clt.labels_, return_counts=True)
        numUniqueFaces = len(np.where(labelIDs > -1)[0])
        print("[INFO] # unique faces: {}\n".format(numUniqueFaces))
 
        return clt.labels_


######################################################################################################
###
###
######################################################################################################



class FaceImageGenerator:
    
    def GenerateImages(self, labels, cluster_dir="/mnt/golem/frodo/", OutputFolderName = "ClusteredFaces", 
                                            MontageOutputFolder = "Montage",
                       file_path= "/mnt/golem/frodo/Database/FaceDB.json",
                       IDs = None
                      ):
        import shutil
        import os
        import time
        import json
        import numpy as np
        import cv2
        from imutils import build_montages
        import taqdm
        
        OutputFolder = cluster_dir + OutputFolderName
 
        if not os.path.exists(OutputFolder):
            os.makedirs(OutputFolder)
        else:
            shutil.rmtree(OutputFolder)
            time.sleep(0.5)
            os.makedirs(OutputFolder)
 
        MontageFolderPath = os.path.join(OutputFolder, MontageOutputFolder)
        os.makedirs(MontageFolderPath)
        
        if not (os.path.isfile(file_path) and os.access(file_path, os.R_OK)):
            print('The input encoding file, ' + str(file_path) + ' does not exists or unreadable')
            exit()
            
        self.DB = json.loads(open(file_path).read())
        
        if IDs != None:
            self.DB = {k: v for k,v in self.DB.items() if k in IDs}
                
        self.IDs = list(self.DB.keys())
        self.encodings = np.vstack([np.array(v["encodings"]) for k,v in self.DB.items()]) 
 
        
        
        
        (labelIDs, labelIDs_freq) = np.unique(labels, return_counts=True)
         
        # loop over the unique face integers
        for labelID, labelID_freq in taqdm(zip(labelIDs, labelIDs_freq)):
            # find all indexes into the `data` array
            # that belong to the current label ID, then
            # randomly sample a maximum of 25 indexes
            # from the set
             
            print("[INFO] There are {} faces for FaceCluster ID: {}".format(labelID_freq, labelID))
 
            FaceFolder = os.path.join(OutputFolder, "Face_" + str(labelID))
            os.makedirs(FaceFolder)
 
            idxs = np.where(labels == labelID)[0]
            
            ID = np.array(self.IDs)[idxs].tolist()
 
            # initialize the list of faces to
            # include in the montage
            portraits = []
 
            # loop over the sampled indexes
            for i in ID:
                 
                # load the input image and extract the face ROI
                image = cv2.imread(self.DB[i]["path_croped_pic"])
 
                portrait = image
 
                if len(portraits) < 25:
                    portraits.append(portrait)
 
                FaceFilename = "face_" + str(i) + ".png"
 
                FaceImagePath = os.path.join(FaceFolder, FaceFilename)
                cv2.imwrite(FaceImagePath, portrait)

            montage = build_montages(portraits, (96, 120), (5, 5))[0]
             
            MontageFilenamePath = os.path.join(
               MontageFolderPath, "FaceCluster_" + str(labelID) + ".jpg")
                
            cv2.imwrite(MontageFilenamePath, montage)
            
################################################################################################################################
###
###
################################################################################################################################

class FaceRecognition:
    def who_is_it(self, image_path, database):
        import numpy as np
        import tensorflow as tf
        """
        Implements face recognition for the office by finding who is the person on the image_path image.

        Arguments:
            image_path -- path to an image
            database -- database containing image encodings along with the name of the person on the image

        Returns:
            min_dist -- the minimum distance between image_path encoding and the encodings from the database
            identity -- string, the name prediction for the person on image_path
        """

        ### START CODE HERE
        ## Step 1: Compute the target "encoding" for the image. Use img_to_encoding() see example above.
        self.import_FaceNet()
        self.encoding =  self.img_to_encoding(image_path)

        ## Step 2: Find the closest encoding ##
        # Initialize "min_dist" to a large value, say 100
        min_dist = 100
        
        # Loop over the database dictionary's names and encodings.
        for (name, db_enc) in database.items():

            # Compute L2 distance between the target "encoding" and the current db_enc from the database.

#             dist = (np.vstack(db_enc["encodings"]) - self.encoding)**2
#             dist = np.sum(dist, axis=1)
#             dist = np.min(np.sqrt(dist))
        
            dist = np.linalg.norm(tf.subtract(np.vstack(db_enc["encodings"]), self.encoding), axis=1, ord=2).min()
           
#             dist = np.array([np.mean(np.array(dist))])

            # If this distance is less than the min_dist, then set min_dist to dist, and identity to name. (≈ 3 lines)
            if dist < min_dist:
                min_dist = dist
                identity = name
                
        if min_dist > 0.75:
            print("Not in the database.")
        else:
            print ("it's " + str(identity) + ", the distance is " + str(min_dist))

        return min_dist, identity
    
    def import_FaceNet(self, model_path='../dat/models/keras-facenet-h5/model.json', 
                       weight_path = '../dat/models/keras-facenet-h5/model.h5'):

        from tensorflow.keras.models import model_from_json
        json_file = open(model_path, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.FaceNet = model_from_json(loaded_model_json)
        self.FaceNet.load_weights(weight_path)
    
    def img_to_encoding(self, image_path):
        import tensorflow as tf  
        import numpy as np
        img = tf.keras.preprocessing.image.load_img(image_path, target_size=(160, 160))
        img = np.around(np.array(img) / 255.0, decimals=12)
        x_train = np.expand_dims(img, axis=0)
        embedding = self.FaceNet.predict_on_batch(x_train)
        return embedding / np.linalg.norm(embedding, ord=2)
    
