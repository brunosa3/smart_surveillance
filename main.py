import argparse
import os
import re
from FaceEncoder import *

def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError(f'{value} is not a valid boolean value')


# define all the arguments you want parese to the python scripts
def parse_args():
    parser = argparse.ArgumentParser(description='arguments to control the pipline')
    
    ########################################################################################################################################
    ### input parsed to main(*args)
    ########################################################################################################################################
    
    # use pictures of all, None or list of specific subfolders
    # so you can specify if you wish to include pictures within subfolder of specified picture directory or you want to stop the pipline to go deeper in subfolders
    parser.add_argument('--subfolders', default = None, help='use pictures of all, None or list of specific subfolders') 
    
    # decide if you want to cluster the the extracted faces
    parser.add_argument('--cluster', type=str_to_bool, nargs='?', const=True, default=False, help='run the clustering pipline')
    
    # decide which image formats you want to include - eg videos are ignored - until now i only considered these image formats
    parser.add_argument('--prefix', default = "jpg|heic|PNG|JPG|HEIC", help='the prefix which files are allowed to use e.g jpg|heic|PNG|JPG|HEIC')   
    
    # decide if you want to archive the processed faces in the database or not    
    parser.add_argument('--archive', type=str_to_bool, nargs='?', const=False, default=True, help='archive the extracted faces')
    
    # this is not implemented yet - therefore its running only on synchronous CPU - the whole pipline is not very efficient
    # the mentioned 56k faces took me 1,5 days to run (~2sec per image)
    # so there is big potential to make it more efficient - but not sure if that is needed as it is run anyway just once before you start the system - after our software in online it will generate its on database
    # parser.add_argument('--gpus', type=int, default=0, help='number of gpus (not implemented yet)') # not yet implemented
        
    ########################################################################################################################################
    ### input parsed to class GenerateFaces(*args)
    ########################################################################################################################################
    
    # the path in which the input pictures are stored
    parser.add_argument('--input_dir', default="/mnt/golem/Photos/Caren_externe_Festplatte/", help='the path in which the input pictures are stored')
    
    # the path in which the extracted/cropped faces should be stored
    parser.add_argument('--output_dir', default="/mnt/golem/frodo/cropped_faces/", help='the path in which the extracted faces should be stored')
    
    # run the pipline with only faces that were detected above the seted threshold - I run it for any detected face (thr=0) such
    # that you can judge yourself after with the clustering - there you could also decide what confidence you want to have for the clustering
    parser.add_argument('--thr', default=0, help='run the pipline with only faces that were detected above the seted threshold')
    
    # in case you would like to see the extracted faces in while the pipline is running - not recommended
    parser.add_argument('--plot', type=str_to_bool, nargs='?', const=True, default=False, help='do you want to get plots?')
    
    
    ########################################################################################################################################
    ### input parsed to GenerateFaces(*args)
    ###                    --> self.import_FaceNet(*args)
    ########################################################################################################################################
    
    # path to the FaceNet config file path
    parser.add_argument('--model_path', default = '../dat/models/keras-facenet-h5/model.json', help='path to the FaceNet model')   
    
    # path to the FaceNet weight file path
    parser.add_argument('--weight_path', default = '../dat/models/keras-facenet-h5/model.h5', help='path to the pretrained weights of the FaceNet model')
    
    ########################################################################################################################################
    ### input parsed to GenerateFaces().run(*args)
    ########################################################################################################################################
    
    # run the face detection, cropping and encoding pipline 
#     parser.add_argument('--run', default = True,  help='run the face detection, cropping and encoding pipline') 
    parser.add_argument('--run', type=str_to_bool, nargs='?', const=False, default=True, help='run the face detection, cropping and encoding pipline')
    
    # I created batches to archive the processed faces into the database (eg if the pipline is breaking after thousand of pictures you don t loose all the processed pictures only till the last batch - here we save all the extracted faces in the database after 100 processed pictures - so you maximal loose 99 pictures in worst case)
    parser.add_argument('--batch_size', default = 100, type=int, help='number of pictures until we save the extracted faces into the database')  
    

    ########################################################################################################################################
    ### input parsed to FaceCluster(*agrs) & FaceImageGenerator().GenerateImages(*args)
    ########################################################################################################################################
    
    # run the clustering pipline only with some specific pictures within a specified folder must be a list of picture filenames
    parser.add_argument('--IDs', default=None, help='run the clustering pipline only with some specific pictures within a specified folder must be a list of picture filenames')
    
    # specify the directory + filename where the database should be stored | don't forget the .json extension 
    parser.add_argument('--DB_path', default="/mnt/golem/frodo/Database/FaceDB.json", help='specify the directory + any filename you would like to call that repository/database to archive the detected faces - do not forget the .json extension')
    
    ########################################################################################################################################
    ### input parsed to FaceImageGenerator().GenerateImages(*args)
    ########################################################################################################################################
    
    # specify the directory in which the clustered faces should be stored
    parser.add_argument('--cluster_dir', default="/mnt/golem/frodo/", help='specify the directory in which the clustered faces should be stored')
    
    # specify the folder name in which the clustered faces should be stored -> will create a new folder with that name
    parser.add_argument('--cluster_folder_name', default="ClusteredFaces", help='specify the folder name in which the clustered faces should be stored')
    
    # specify the folder name in which the montage faces per cluster should be stored (25 faces/cluster) -> will create a new folder with that name
    parser.add_argument('--montage_folder_name', default="Montage", help='specify the folder name in which the montage faces per cluster should be stored (25 faces/cluster)')
    
    ########################################################################################################################################
    ### input parsed to GenerateFaces.restructureDB(*args)
    ########################################################################################################################################
    
    # last step of pipline - restructuring the database such that we have human meaningfull subject profiles including real subject names
    parser.add_argument('--restructureDB', type=str_to_bool, nargs='?', const=True, default=False,  help='define if you want to restructure the database')
        
    # specify the path to the manually checked clustered path - each folder contains faces of the same subject and the folder name is the name of the subject
    parser.add_argument('--cluster_path', default="/mnt/golem/frodo/ClusteredFaces_clean/", help='specify the path to the manually checked clustered path - each folder contains faces of the same subject and the folder name is the name of the subject')
    
    # specify the path + filename of the new restructured database
    parser.add_argument('--new_DB_path', default="/mnt/golem/frodo/Database/New_FaceDB.json", help='specify the path + filename of the new restructured database')
    
    # a list of folders that should be ignored for example the aste busket folderfrom HDBSCAN Face_-1
    parser.add_argument('--ignore', default=["Face_-1", "Montage"], nargs="+", help='a list of folders that should be ignored for example the aste busket folderfrom HDBSCAN Face_-1')
    
    # True if you want to ignore any cluster folder within the directory that starts with Face_* - hinting to folders that are still unlabeled/unidentified
    parser.add_argument('--ig_faces', type=str_to_bool, nargs='?', const=False, default=True,  help='True if you want to ignore any cluster folder within the directory that starts with Face_* - hinting to folders that are still unlabeled/unidentified')
        
   
    # parse all specified arguments & return it to main script
    args = parser.parse_args()

    return args




# that is the pipline backbone - deciding what steps to trigger
def main():
    # get all the parsed arguments
    args = parse_args()
    
    # check if we need to consider all sub-folders within specified directory    
    if args.subfolders == "all":
        # walk through each subfolder till you reach a filename in list comprehension style
        pictures = [os.path.join(root, name)
        for root, dirs, files in os.walk(args.input_dir)
            for name in files
                if name.endswith((tuple(i for i in args.prefix.split("|"))))]
        
        # for me i found files that included @ signs which I had to filter out - don t know if thats true for you too
        # seemed like synology sys files that are created for shared folders
        pictures = [i for i in pictures if "@" not in i]
        
    # only use the pictures which are in the specified folder (not sub-folders)    
    elif args.subfolders == None:
        
        # extract all the paths to the images in the specified folder
        pictures = [args.input_dir + IMG  for IMG in os.listdir(args.input_dir)]          
    
    # look in specified sub-folders
    else:   
        
        # identify all the subfolders within the specified directory
        # this could be done more elegant with the os.path.isdir() and os.access(file_path, os.R_OK) functions
        folders = {args.input_dir + w: os.listdir(args.input_dir + w) for w in os.listdir(args.input_dir) if (len(w.split(".")) == 1) and ("@" not in w)}
        
        # extract all the paths to the pictures within the specified subfolders
        pictures = [k + "/" + v for k in folders.keys() if k in args.subfolders for v in folders[k]]
    
    # regex matching all pictures with the specified prefix-ignoring everything else
    r = re.compile(".*(" + args.prefix + ")")
    pictures = list(filter(r.match, pictures ))
    
    print("[INFO] preparing pipline")
    
    
    # initiate the pipline with the specified arguments
    DB = GenerateFaces(file_path=pictures, output=args.output_dir, plot=args.plot, thr=args.thr, model_path=args.model_path, weight_path=args.weight_path, database=args.DB_path)
    
    # decide if we need to run the face extraction and encoding pipline
    if args.run:
        print("[INFO] starting to extract faces faces from {} pictures".format(len(pictures)))
        print("[INFO] specified input directory {}, output_directory {}, plot {}, confidence threshold {}".format(args.input_dir, args.output_dir, args.plot, args.thr))
        print("[INFO] you want to archive the extractions {}, batch_size {}".format(args.archive, args.batch_size))
        DB.run(archive=args.archive, batch_size=args.batch_size)
    
    # decide if we need to run the clustering pipline
    if args.cluster:
        print("[INFO] preparing clustering of extracted faces")
        print("[INFO] specified database directory {}, specified IDs {}".format(args.DB_path, args.IDs))
        
        # cluster the extracted faces  
        labelIDs = FaceCluster(IDs = args.IDs, file_path=args.DB_path).Cluster()  
        
        # output the clusterings in folders of specified folder names 
        FaceImageGenerator().GenerateImages(labels=labelIDs, file_path= args.DB_path, OutputFolderName =args.cluster_folder_name, MontageOutputFolder = args.montage_folder_name, IDs=args.IDs) 
        
    if args.restructureDB:
        print("[INFO] start to restructure the database")
        new_DB = DB.restructure_DB(cluster_path= args.cluster_path,
                          new_DB_path = args.new_DB_path,
                          ignore= args.ignore,
                          ig_faces= args.ig_faces)
        
        print("[INFO] There are {} subjects in the reference database".format(len(new_DB)))


if __name__ == '__main__':
    main()