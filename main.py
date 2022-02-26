import argparse
import os
import re
from FaceEncoder import *

def parse_args():
    parser = argparse.ArgumentParser(description='arguments to control the pipline')
    
    # GenerateFaces()
    parser.add_argument('--input_dir', default="/mnt/golem/Photos/Caren_externe_Festplatte/", help='the path in which the input pictures are stored')
    parser.add_argument('--output_dir', default="/mnt/golem/frodo/cropped_faces/", help='the path in which the extracted faces should be stored')
    parser.add_argument('--thr', default=0, help='run the pipline with only faces that were detected above the seted threshold')
    parser.add_argument('--plot', default=False, type=bool, help='run the pipline with only faces that were detected above the seted threshold')

    # .run()
    parser.add_argument('--run', default = True, type=bool, help='run the face detection, cropping and encoding pipline')    
    parser.add_argument('--batch_size', default = 100, type=int, help='number of pictures until we save the extracted faces into the database')  
    
    # import_FaceNet()
    parser.add_argument('--model_path', default = '../dat/models/keras-facenet-h5/model.json', help='path to the FaceNet model')    
    parser.add_argument('--weight_path', default = '../dat/models/keras-facenet-h5/model.h5', help='path to the pretrained weights of the FaceNet model')
    
    # FaceCluster() & FaceImageGenerator().GenerateImages()
    parser.add_argument('--IDs', default=None, help='run the clustering pipline only with some specific pictures must be a list')
    #  FaceImageGenerator().GenerateImages()
    parser.add_argument('--cluster_dir', default="/mnt/golem/frodo/", help='specify the directory in which the clustered faces should be stored')
    parser.add_argument('--cluster_folder_name', default="ClusteredFaces", help='specify the folder name in which the clustered faces should be stored')
    parser.add_argument('--montage_folder_name', default="Montage", help='specify the folder name in which the montage faces per cluster should be stored (25 faces/cluster)')
    #  FaceImageGenerator().GenerateImages() & FaceCluster()
    parser.add_argument('--DB_path', default="/mnt/golem/frodo/Database/FaceDB.json", help='specify the directory + filename where the database shouldbe stored')

    
    # main()
    parser.add_argument('--subfolders', default = None, help='use pictures of all, None or list of specific subfolders') 
    
    parser.add_argument('--cluster', default=False, type=bool, help='run the clustering pipline')
    
    parser.add_argument('--prefix', default = "jpg|heic|PNG|JPG|HEIC", help='the prefix which files are allowed to use e.g jpg|heic|PNG|JPG|HEIC')   
    
    parser.add_argument('--archive', default=True, help='archive the extracted faces')
    
    # parser.add_argument('--gpus', type=int, default=0, help='number of gpus (not implemented yet)') # not yet implemented
   
    args = parser.parse_args()

    return args


def main():
    # get all the parsed arguments
    args = parse_args()
    # check if we need to consider all sub-folders within specified directory
    
    if args.subfolders == "all":
        # walk through each subfolder till you reach a filename
        pictures = [os.path.join(root, name)
        for root, dirs, files in os.walk(args.input_dir)
            for name in files
                if name.endswith((tuple(i for i in args.prefix.split("|"))))]

        pictures = [i for i in pictures if "@" not in i]
        
    # only use the pictures which are in the specified folder (not sub-folders)    
    elif args.subfolders == None:
        
        pictures = [args.input_dir + IMG  for IMG in os.listdir(args.input_dir)]          
    
    # look in specified sub-folders
    else:   
        
        # identify all the subfolders within the specified directory
        folders = {args.input_dir + w: os.listdir(args.input_dir + w) for w in os.listdir(args.input_dir) if (len(w.split(".")) == 1) and ("@" not in w)}
        # extract all the paths to the pictures within the specified subfolders
        pictures = [k + "/" + v for k in folders.keys() if k in args.subfolders for v in folders[k]]
    
    # regex matching all pictures with the specified prefix-ignoring everything else
    r = re.compile(".*(" + args.prefix + ")")
    pictures = list(filter(r.match, pictures ))
    
    print("[INFO] preparing pipline to extract faces from {} pictures".format(len(pictures)))
    print("[INFO] specified input directory {}, output_directory {}, plot {}, confidence threshold {}".format(args.input_dir, args.output_dir, args.plot, args.thr))
    
    DB = GenerateFaces(file_path=pictures, output=args.output_dir, plot=args.plot, thr=args.thr)

    if args.run:
        print("[INFO] starting to extract faces") 
        print("[INFO] you want to archive the extractions {}, batch_size {}".format(args.archive, args.batch_size))
        DB.run(archive=args.archive, batch_size=args.batch_size)
    
    if args.cluster:
        print("[INFO] preparing clustering of extracted faces")
        print("[INFO] specified database directory {}, specified IDs {}".format(args.DB_path, args.IDs))
        labelIDs = FaceCluster(IDs = args.IDs, file_path=args.DB_path).Cluster()        
        FaceImageGenerator().GenerateImages(labels=labelIDs, file_path= args.DB_path, OutputFolderName =args.cluster_folder_name, MontageOutputFolder = args.montage_folder_name, IDs=args.IDs) 


if __name__ == '__main__':
    main()