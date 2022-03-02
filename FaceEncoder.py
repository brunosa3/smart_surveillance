class GenerateFaces: 
    """
    A class to extract and encode faces of specified pictures
    
    ...
        
    Description
    ----------
    GenerateFaces is a class with multiple sub functions which are trying to detect faces in any given picture using the Multi-task 
    Cascaded Convolutional Networks (MTCNN). The process of MTCNN consists of three tasks (Face classification, Bounding box regression and
    Facial Landmark localization) of convolutional networks that are able to recognize faces and landmark location such as eyes, nose, 
    and mouth. We then recenter the detected faces using the determined landmarks (eyes, node, mouth) because we need to make sure that
    width and height of the cropped faces are of equal size as input to our FaceRecognition model (required input 160x160). 
    Next we crop the faces out of the pictures and use the last layer of our face recognition model (here FaceNet) to encode the cropped
    faces and archive all the intermediate and final results in a .json database
    
    ...

    Methods
    -------
    run()
        run the pipline
    get_database()
        returns the full database

    ...

    Helper functions
    -------
    crop_faces_centered_boxes_for_encoding()
        iterate through each detected face, recenter, crop, encode and archive (if requested also plot) each face
    import_FaceNet()
        import FaceNet
    img_to_encoding()
         scales the image to expected input shape (160x160) and runs the forward propagation of the model on the specified image to get 
         the encodings of the faces
    archive_database()
        archive the generated database to specified output directory    
        
    """
        
    # The init method or constructor
    def __init__(self, file_path , output="/mnt/golem/frodo/cropped_faces/", thr=0, plot=True, model_path='../dat/models/keras-facenet-h5/model.json', weight_path = '../dat/models/keras-facenet-h5/model.h5', database="/mnt/golem/frodo/Database/FaceDB.json"):
        
        """
        Parameters
        ----------
        file_path : str
            path to the input pictures - pictures to extract faces. If the argument is specified with None. you could use the class to
            import the alredy existing database e.g GenerateFaces(file_path=None, *args).get_database()
        output : str
            path to the output directory - where the cropped faces are stored
        thr : float, optional
            the confidence threshold - probability that the detected face is a face (default 0)
        plot : bool
            if we want to plot the cropped faces in the terminal - not recommended
        model_path : str
            path to the Face Recognition model configuration (FaceNet)
        weigth_path : str
            path to the pretrained weights of the Face Recognition model (FaceNet)
            
        """
            
        import os
        import json
        
        # check if that specified database path already exist
        if os.path.isfile(database):
            # load the pre-existing database
            self.DB = json.loads(open(database).read())
            # determine which unique keys are alredy reserved
            self.n = len(self.DB) 
        
        # if there is no pre-existing database
        else:
            # initiate a container for the datbase and the unique ID
            self.DB = None
            self.n = 0
        
        ### initiate all specified arguments such that they are reachable for sub functions
        
        # path to the input pictures - pictures to extract faces        
        self.path = file_path        
        # path to the output directory - where the cropped faces are stored
        self.output_dir = output
        # the confidence threshold - probability that the detected face is a face
        self.thr = thr
        # if we want to plot the cropped faces in the terminal - not recommended
        self.plot = plot
        # self.container serves as our internal database
        self.container = {}
        # import our Face recognition model to encode the cropped faces
        self.import_FaceNet(model_path=model_path, weight_path=weight_path)
        # initiate the batch number - number of pictures which where alredy processed in a certain batch
        self.batch = 0
#         self.DB = self.get_existingDB(self, path=None)

        
    def run(self, archive = True, batch_size = 100):
        
        """
        starts the pipline.
        
        ...

        Description
        ----------
        .run() launch the pipline by looping through each specified image detect if and where there is a face (MTCNN), recenter 
        (using landmarks in the face), crop, encode (FaceNet) and archive the intermediate as well as final results
        
        ...
       
        Parameters
        ----------
        archive : bool, optional
            determines if you want to archive the intermediate and final results into the specified database (default is True)
            
        batch_size : int, optional
        the number of processed pictures after which the database will be loged (default is 100)
        
        ...
        
        Output
        ----------
        1) archives the intermediate and final results in a json (our reference database used for clustering) file with the following
        structure
        
        uniqueID: 
            path_pic_orig : str - path of original image
            path_croped_pic : str - path of cropped image/face
            encodings: list - last layer of FaceNet (1x128)
            recentered_rectangle: list - coordinates of cropped recentered face in original image in form of [x1, x2, y1, y2]
            detector : dic - MTCNN output containing Face Classification, Bounding Box Regression and Facial Landmark Localization  
                box : list - coordinates of original non-recentered detected face in original image in form of [x1, y1, width, height]
                confidence : float - probability that the given rectangle is a real face 
                keypoints : dic - landmark coordinates 
                    left_eye :  tuple of x,y ccoordinates marking the left eye
                    right_eye : tuple of x,y ccoordinates marking the right eye
                    nose : tuple of x,y ccoordinates marking the nose eye
                    mouth_left : tuple of x,y ccoordinates marking the left mouth
                    mouth_right : tuple of x,y ccoordinates marking the right mouth
                    
        2) each detected face will be cropped out of the original image and saved in the specified output directory in the following
        structure
        
        if the probability is higher than the specified threshold (default = 0)        
            uniqueID_cropped_face.png: file in specified ouput directory        
        otherwise
            less_confident_faces: folder
                {}(confidence)_cropped_face.png: file in sub of output directory     


        """
        
        from tqdm import tqdm 
        import matplotlib.pyplot as plt
        from mtcnn.mtcnn import MTCNN
        from PIL import Image
        from pillow_heif import register_heif_opener
        import numpy as np

        # loop throug each image                
        for pics in tqdm(self.path):
            
            if self.DB != None:
                # check if the image was already processed and successfully archived in database
                if pics in list(set([v["path_pic_orig"] for v in self.DB.values()])):
                    print("[INFO] this image is already in the Database {}".format(pics))
                    continue
            
            # save the image path such that it can be reach in helper/sub functions
            self.filename = pics

            # load the image depending on image format/prefix
            if ("HEIC" == self.filename.split(".")[-1]) | ("heic"== self.filename.split(".")[-1]): #could also use .lower() to have in one go
                # HEIC is Apple’s proprietary version of the HEIF or High-Efficiency Image File format. This newer file format is 
                # intended to be a better way to save your pictures, making your images smaller in terms of data while retaining high
                # quality.

                register_heif_opener()

                try:
                    image = Image.open(self.filename)
                    self.pixels = image.__array__()                
                except Exception as e:
                    print("[WARNING] there was an error in this image {} - maybe it is truncated?".format(self.filename))
                    print(e)
                    continue
                
                
            elif ("PNG" == self.filename.split(".")[-1]) | ("png"== self.filename.split(".")[-1]):
                
                # png have 4 channels R,G,B and alpha for transparency --> here we get rid of the aloha/transperency channel
                try:
                    image = Image.open(self.filename).convert('RGBA')
                    background = Image.new('RGBA', image.size, (255,255,255))
                    alpha_composite = Image.alpha_composite(background, image)
                    alpha_composite_3 = alpha_composite.convert('RGB')
                    
                except Exception as e:
                    print("[WARNING] there was an error in this image {} - maybe it is truncated?".format(self.filename))
                    print(e)
                    continue
                
                
                self.pixels = np.asarray(alpha_composite_3)

            else:
                try:
                    # here we try to import any other image format - designed for prefix jpg|JPG - but who knows maybe that is also valid for other formats - did not test yet
                    self.pixels = plt.imread(self.filename)
                
                except Exception as e:
                    # if there was an error although its in jpg format - its probably truncated
                    if ("jpg" == self.filename.split(".")[-1].lower()):
                        print("[WARNING] there was an error in this image {} - maybe it is truncated?".format(self.filename))
                                               
                    else:
                        # otherwise its probably because we did not integrate the specified format
                        print("[WARNING] there was an error in this image {} - We do not support this image format yet {}".format(self.filename, self.filename.split(".")[-1]))
                        
                    print(e)   
                    continue
                
                
            # create/initiate the detector, using default weights
            detector = MTCNN()
            # detect faces in the image
            self.faces = detector.detect_faces(self.pixels)
            # crop, recenter and encode the detected faces
            self.crop_faces_centered_boxes_for_encoding()            
            
            # archive the temporaly saved processed faces when ever the batch size is full 
            if archive: 
                if self.batch > batch_size:
                    print("[INFO] archive batch")
                    self.archive_database()
                    self.batch = 0
                else:
                    self.batch += 1
            

        # end statement with some statistics when run completed 
        if archive:
            print("[INFO] preparing final archive of FaceDB ... \nlooped through {} pictures, detected {} faces in {} pictures".format(len(self.path), len(self.container), len(np.unique([v["path_pic_orig"] for v in self.container.values()]))))
            self.archive_database()
            print("[INFO] archived 'FaceDB' sucessfully!!!")
            print("[INFO] There are currently {} faces in FaceDB encoded".format(len(self.container)))
            
        else:
            print("[INFO] DB was not archived")
        
    # draw each face separately
    def crop_faces_centered_boxes_for_encoding(self):
        import cv2
        import matplotlib.pyplot as plt
#         import uuid
        import numpy as np
        import os
        
        # input dimension
        dim = self.pixels.shape
        
        # loop through each detected face in a picture - remember one image can have multiple faces - think about group photos and so on
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
                    
                
                # get coordinates of original detected face 
                x1, y1, width, height = self.faces[i]['box']
                x2, y2 = x1 + width, y1 + height

                # recenter the detected rectangle around the face
                circles = self.faces[i]['keypoints'].values() # rember keypoints contain dic of landmarks in format of tuple(x,y)
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


                # crop the face out of the image 
                crop = self.pixels[y1:y2, x1:x2]
                
                # save the cropped face in in specified output directory for manual inspection
                output_dir = self.output_dir + str(ID) + "_cropped_face.png"
                cv2.imwrite(output_dir, cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
                
                # resize and encode the faces 
                self.encodings = self.img_to_encoding(output_dir)
                
                # check if there is alredy an existing database
                if self.DB != None:
                    # check if we have alredy processed this image by checking the encodings
                    # note at the beginning of the code we alredy checked if that image was processed. However, this was just based on the 
                    # filename - we could rename the image and have it as unique image - thats not what we want 
                    # therefore i implemented this part to be sure that we have only unique faces with no replications
                    match = [k for k,v in self.DB.items() if v["encodings"] == self.encodings.tolist()]
                    if len(match) > 0:
                        print("[INFO] This face already existed in the database {}".format(match[0]))
                        # since we have alredy saved this cropped file before we need to remove it from the directory
                        os.remove(output_dir)
                    
                    # if this face was not already stored in the database archive it temproaly until the batch size is full filled 
                    else:
                        self.container[ID] = {"path_pic_orig": self.filename,
                                              "path_croped_pic": output_dir,
                                     "encodings": self.encodings.tolist(),
                                     "recentered_rectangle": [x1, x2, y1, y2],
                                     "detector": self.faces[i]}
                # if there is no pre-existing database archive the intermediate and final results temproaly until the batch size is full filled
                else:                
                    self.container[ID] = {"path_pic_orig": self.filename,
                                          "path_croped_pic": output_dir,
                                     "encodings": self.encodings.tolist(),
                                     "recentered_rectangle": [x1, x2, y1, y2],
                                     "detector": self.faces[i]}
                
                # plot the faces if wanted - not recommended
                if self.plot:
                    # define subplot
                    plt.subplot(1, len(self.faces), i+1)
                    plt.axis('off')
                    # plot face
                    plt.imshow(crop)
            
            
            # if confident in face detection is too low < thr 
            # skip the processing part and just save the original detected face by MCTNN and save the cropped face into a sub folder
            # called less_confident_faces in output directory
            else:
                # get coordinates of original detected face of MCTNN
                x1, y1, width, height = self.faces[i]['box']
                x2, y2 = x1 + width, y1 + height
                
                #check if the folder less_confident_faces is alredy available if not -> create the folder
                if os.path.isdir(self.output_dir + "less_confident_faces") == False:
                    os.mkdir(self.output_dir + "less_confident_faces")
                
                # save the cropped detected face
                output_dir = self.output_dir + "less_confident_faces/" + str(self.faces[i]['confidence']) + "_cropped_face.png"    
                cv2.imwrite(output_dir, cv2.cvtColor(self.pixels[y1:y2, x1:x2], cv2.COLOR_RGB2BGR))
                
                print("[INFO] This face did not had high confidence {}".format(self.faces[i]['confidence']))
                
                
    def import_FaceNet(self, model_path, weight_path):
        
        """
        import FaceNet 
        
        ...

        Description
        ----------
        .importFaceNet() imports the FaceNet model. FaceNet is a face recognition system developed in 2015 by researchers at Google that
        achieved then state-of-the-art results on a range of face recognition benchmark datasets.
        
        ...
       
        Parameters
        ----------
        model_path : str
            path to the FaceNet config file path
            
        weight_path : str
            path to the FaceNet weight file path

        """

        from tensorflow.keras.models import model_from_json
        
        json_file = open(model_path, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.FaceNet = model_from_json(loaded_model_json)
        self.FaceNet.load_weights(weight_path)

    #tf.keras.backend.set_image_data_format('channels_last')
    def img_to_encoding(self, image_path):
        """
        scales the image to expected input shape (160x160) and runs the forward propagation of the model on the specified image to get 
        the encodings of the faces 
        
        ...

        Description
        ----------
        .img_to_encoding() scales the image to expected input shape (160x160) for the FaceNet model and runs the forward propagation of the
        model on the specified image to get the encodings of the faces
        
        ...
       
        Parameters
        ----------
        image_path : str
            path to the image file
            
            
        Output
        ----------
        encoding : np.array of shape (1,128)
            last layer of FaceNet model

        """
        import tensorflow as tf  
        import numpy as np
        img = tf.keras.preprocessing.image.load_img(image_path, target_size=(160, 160))
        img = np.around(np.array(img) / 255.0, decimals=12)
        x_train = np.expand_dims(img, axis=0)
        embedding = self.FaceNet.predict_on_batch(x_train)
        return embedding / np.linalg.norm(embedding, ord=2)
   
    
    def archive_database(self):
        """
        archive the database
        
        ...

        Description
        ----------
        .archive_database() converts and archives the temporaly stored dictonary to a json file in the specified 
        directory. Containg all intermediate as well as final results of run() pipline in the format format.
        
        uniqueID: 
            path_pic_orig : str - path of original image
            path_croped_pic : str - path of cropped image/face
            encodings: list - last layer of FaceNet (1x128)
            recentered_rectangle: list - coordinates of cropped recentered face in original image in form of [x1, x2, y1, y2]
            detector : dic - MTCNN output containing Face Classification, Bounding Box Regression and Facial Landmark Localization  
                box : list - coordinates of original non-recentered detected face in original image in form of [x1, y1, width, height]
                confidence : float - probability that the given rectangle is a real face 
                keypoints : dic - landmark coordinates 
                    left_eye :  tuple of x,y ccoordinates marking the left eye
                    right_eye : tuple of x,y ccoordinates marking the right eye
                    nose : tuple of x,y ccoordinates marking the nose eye
                    mouth_left : tuple of x,y ccoordinates marking the left mouth
                    mouth_right : tuple of x,y ccoordinates marking the right mouth
        
        ...
                   
        Output
        ----------
        database : file
            json file containg all intermediate as well as final results of run() pipline

        """
        import json
        DB = self.get_database()        
        with open('/mnt/golem/frodo/Database/FaceDB.json', 'w') as fp:
            json.dump(DB, fp)
            
    def get_database(self):
        """
        get the database
        
        ...

        Description
        ----------
        .get_database() checks if there is already a pre-existing database an returns the full database (=pre-existing DB + generated 
        DB). Database has the following format. 
        
        uniqueID: 
            path_pic_orig : str - path of original image
            path_croped_pic : str - path of cropped image/face
            encodings: list - last layer of FaceNet (1x128)
            recentered_rectangle: list - coordinates of cropped recentered face in original image in form of [x1, x2, y1, y2]
            detector : dic - MTCNN output containing Face Classification, Bounding Box Regression and Facial Landmark Localization  
                box : list - coordinates of original non-recentered detected face in original image in form of [x1, y1, width, height]
                confidence : float - probability that the given rectangle is a real face 
                keypoints : dic - landmark coordinates 
                    left_eye :  tuple of x,y ccoordinates marking the left eye
                    right_eye : tuple of x,y ccoordinates marking the right eye
                    nose : tuple of x,y ccoordinates marking the nose eye
                    mouth_left : tuple of x,y ccoordinates marking the left mouth
                    mouth_right : tuple of x,y ccoordinates marking the right mouth
        
        ...
                   
        Output
        ----------
        database : file
            json file containg all intermediate as well as final results of run() pipline

        """
        if self.DB != None:
            self.container.update(self.DB)            
        return self.container
    
    def restructure_DB(self,
                       cluster_path="/mnt/golem/frodo/ClusteredFaces_clean/",
                       new_DB_path = "/mnt/golem/frodo/Database/New_FaceDB.json",
                       ignore=["Face_-1"],
                       ig_faces=True):
            import os
            import re
            import json
            from tqdm import tqdm
            # get the raw database
            FaceDB = self.DB
            # get all the identities of user defined labels (=folder names)
            ID = os.listdir(cluster_path)
            
            ID = [id for id in ID if os.path.isdir(cluster_path + id)] 
            
            if ig_faces:
                ignore += [id for id in ID if re.match("Face.*", id)]

            # get all the unique keys of the corpped faces cluster together
            r = re.compile(".*png")
            pictures = {id: [re.split("[_.]",v)[1] for v in list(filter(r.match, os.listdir(cluster_path + id)))]  
                        for id in ID 
                        if id not in ignore}
            # restructure the DB to the form of {real person name: list of FaceDB dict of the identified faces}
            p_o = []
            p_c = []
            e = []
            r_r = []
            de = []    
            New_DB = {}
            for new_id, old_id in tqdm(pictures.items()):
                for k in old_id:
                    for K,V in FaceDB[k].items():
                        if K == 'path_pic_orig':
                                p_o.append(V) 
                        elif K == 'path_croped_pic':
                            p_c.append(V)
                        elif K == 'encodings':
                            e.append(V)
                        elif K == 'recentered_rectangle':
                            r_r.append(V)
                        else:
                            de.append(V)
                New_DB.update({new_id: {'path_pic_orig': p_o, 
                                   'path_croped_pic': p_c, 
                                   'encodings': e, 
                                   'recentered_rectangle': r_r, 
                                   'detector': de}})
                p_o = []
                p_c = []
                e = []
                r_r = []
                de = [] 

            with open(new_DB_path, 'w') as fp:
                json.dump(New_DB, fp)

            return New_DB

                


######################################################################################################
###
###
######################################################################################################

class FaceCluster:
    """
    A class to cluster the extracted faces based on the encodings of the cropped faces.
    
    ...
        
    Description
    ----------
    FaceCluster is a class to cluster the extracted faces based on the encodings of the cropped faces. This class uses the improved DBSCAN 
    method called HDBSCAN for clustering
    
    ...

    Methods
    -------
    Cluster()
        run the pipline
        
    ...
    
    Output
    -------
    cluster labels of HDBSCAN
    
        
    """
     
    def __init__(self, file_path="/mnt/golem/frodo/Database/FaceDB.json", IDs = None):
        
        """
        Parameters
        ----------
        file_path : str
            path to the database you have just created using GenerateFaces().run()  
        IDs : list
            run the clustering pipline only with a selection of faces within a specified folder must be a list of uniqueID pointing to 
            specific faces
            
        """
        
        import json
        import os
        import numpy as np
        
        # check if there is a pre-existing database
        if not (os.path.isfile(file_path) and os.access(file_path, os.R_OK)):
            print('[ERROR] The input encoding file, ' + str(path) + ' does not exists or unreadable')
            exit()
        
        # load the database
        self.DB = json.loads(open(file_path).read())
        
        # if you specified specific images via the IDs argument consider only these images for further clustering analysis
        if IDs != None:
            self.DB = {k: v for k,v in self.DB.items() if k in IDs}
        
        # generate a matrix of all encodings where each row is a individual face and columns are encodings with dimensions (n,128) 
        self.encodings = np.vstack([np.array(v["encodings"]) for k,v in self.DB.items()]) 
     
    def Cluster(self):
        
        """
        .Cluster() runs HDBSCAN on the face encodings of FaceNet last layer to identify and cluster similar/same faces.
        
        ...
        
        Description
        ----------
        .Cluster() runs HDBSCAN on the face encodings of FaceNet last layer.
        HDBSCAN - Hierarchical Density-Based Spatial Clustering of Applications with Noise. Performs DBSCAN over varying epsilon values
        and integrates the result to find a clustering that gives the best stability over epsilon. This allows HDBSCAN to find clusters of 
        varying densities (unlike DBSCAN), and be more robust to parameter selection.
        
        In practice this means that HDBSCAN returns a good clustering straight away with little or no parameter tuning -- and the primary
        parameter, minimum cluster size, is intuitive and easy to select.
        
        HDBSCAN is ideal for exploratory data analysis; it's a fast and robust algorithm that you can trust to return meaningful clusters 
        (if there are any).
        
        ...
        
        Output
        ----------
        cluster labels of HDBSCAN
        
        
        """
#         from sklearn.cluster import DBSCAN
        import hdbscan
        import numpy as np
 
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
    """
    A class that archives the clustered faces into same cluster folders to get a structured profile for each subject that was detected in 
    the provided images.
    
    ...
        
    Description
    ----------
    FaceImageGenerator is a class that archives the clustered faces into same cluster folders to get a structured profile for 
    each subject that was detected in the provided images. This facilitates the evaluation of correct and wrong clusters.
    
    ...

    Methods
    -------
    GenerateImages()
        archive the clustered faces in structured/clustered folders
           
        
    """
    
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
        from tqdm import tqdm
        
        # create the output folder for the Clustered faces
        OutputFolder = cluster_dir + OutputFolderName
        
        # if that folder is not existing already create the folder 
        if not os.path.exists(OutputFolder):
            os.makedirs(OutputFolder)
        else:
            # If the folder exists, delete the folder and all files and subdirectories below it. 
            shutil.rmtree(OutputFolder)
            # wait a bit before you recreate that folder
            time.sleep(0.5)
            os.makedirs(OutputFolder)
         
        # create the Montage folder path
        MontageFolderPath = os.path.join(OutputFolder, MontageOutputFolder)
        os.makedirs(MontageFolderPath)
        
        # check if there is a exisiting database - if not exit
        if not (os.path.isfile(file_path) and os.access(file_path, os.R_OK)):
            print('The input encoding file, ' + str(file_path) + ' does not exists or unreadable')
            exit()
        
        # otherwise load the database
        self.DB = json.loads(open(file_path).read())
        
        # select only the faces which you clustered in FaceCluster
        if IDs != None:
            self.DB = {k: v for k,v in self.DB.items() if k in IDs}
        
        # pull all the encodings from the database 
        self.IDs = list(self.DB.keys())
        self.encodings = np.vstack([np.array(v["encodings"]) for k,v in self.DB.items()]) 
 
        # get all unique clusteres as well as frequency 
        (labelIDs, labelIDs_freq) = np.unique(labels, return_counts=True)
         
        # loop over the unique face integers
        for labelID, labelID_freq in tqdm(zip(labelIDs, labelIDs_freq)):
            # from the set             
            print("[INFO] There are {} faces for FaceCluster ID: {}".format(labelID_freq, labelID))
            
            # create a folder for each unique cluster  
            FaceFolder = os.path.join(OutputFolder, "Face_" + str(labelID))
            os.makedirs(FaceFolder)
             
            # find all indexes into the `data` array that belong to the current label ID
            idxs = np.where(labels == labelID)[0]            
            ID = np.array(self.IDs)[idxs].tolist()
 
            # initialize the list of faces to include in the montage
            portraits = []
 
            # loop over the sampled indexes
            for i in ID:
                 
                # load the input image
                image = cv2.imread(self.DB[i]["path_croped_pic"])
 
                portrait = image
                # append the montage portrait if the montage isnot full < 25 
                if len(portraits) < 25:
                    portraits.append(portrait)
                
                # assign a face cluster name of the face
                FaceFilename = "face_" + str(i) + ".png"
                
                # archive the individual clustered face 
                FaceImagePath = os.path.join(FaceFolder, FaceFilename)
                cv2.imwrite(FaceImagePath, portrait)
            # build the montage with the appended images of each cluster with at most 25 faces
            montage = build_montages(portraits, (96, 120), (5, 5))[0]
            
            #archive the montagee/portait in specified directory
            MontageFilenamePath = os.path.join(
               MontageFolderPath, "FaceCluster_" + str(labelID) + ".jpg")                
            cv2.imwrite(MontageFilenamePath, montage)
            
################################################################################################################################
###
###
################################################################################################################################

class FaceRecognition:
    def who_is_it(self, image_path, database, model_path='../dat/models/keras-facenet-h5/model.json', weight_path = '../dat/models/keras-facenet-h5/model.h5'):
        import numpy as np
        import tensorflow as tf
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
        self.import_FaceNet(model_path=model_path, weight_path=weight_path)
        # Compute the target "encoding" for the image
        self.encoding =  self.img_to_encoding(image_path)

        # Find the closest encoding 
        # Initialize "min_dist" to a large value, say 100
        min_dist = 100
        
        # Loop over the database dictionary's names and encodings.
        for (name, db_enc) in database.items():
            # Compute L2 distance between the target "encodings" and the current db_enc from the database.
            # note we compute multiple distances for a subject and pick the distance with the minimal distance

#             dist = (np.vstack(db_enc["encodings"]) - self.encoding)**2
#             dist = np.sum(dist, axis=1)
#             dist = np.min(np.sqrt(dist))
        
            dist = np.linalg.norm(tf.subtract(np.vstack(db_enc["encodings"]), self.encoding), axis=1, ord=2).min()
           
#             dist = np.array([np.mean(np.array(dist))])

            # If this distance is less than the min_dist, then set min_dist to dist, and identity to name
            if dist < min_dist:
                min_dist = dist
                identity = name
        
        # if any of the distances is higher than 0.75 we don t think the given face is represented in the database
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
    
