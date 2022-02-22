import argparse
import os
import re
from FaceEncoder import *

def parse_args():
    parser = argparse.ArgumentParser(description='arguments to control the pipline')
    
    parser.add_argument('--subfolders', default = None, help='use pictures of all, None or list of specific subfolders') 
    parser.add_argument('--run', default = True, help='run the face detection, cropping and encoding pipline')    
    parser.add_argument('--prefix', default = "jpg|heic|PNG|JPG|HEIC", help='the prefix which files are allowed to use e.g jpg|heic|PNG|JPG|HEIC')
    parser.add_argument('--input_dir', default="/mnt/golem/Photos/Caren_externe_Festplatte/", help='the path in which the input pictures are stored')
    parser.add_argument('--cluster', default=False, help='run the clustering pipline')
    parser.add_argument('--IDs', default=None, help='run the clustering pipline only with some specific pictures must be a list')
    parser.add_argument('--thr', default=0, help='run the pipline with only faces that were detected above the seted threshold')
    
    parser.add_argument('--gpus', type=int, default=0, help='number of gpus (not implemented yet)')
   
    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    if args.subfolders == "all":
        pictures = [os.path.join(root, name)
        for root, dirs, files in os.walk(args.input_dir)
            for name in files
                if name.endswith((tuple(i for i in args.prefix.split("|"))))]

        pictures = [i for i in pictures if "@" not in i]
        
        #folders = {args.input_dir + w: os.listdir(args.input_dir + w) for w in os.listdir(args.input_dir) if (len(w.split(".")) == 1) and ("@" not in w)}
    
        #pictures = [k + "/" + v for k in folders.keys() for v in folders[k]]
        
    elif args.subfolders == None:
        
        pictures = [args.input_dir + IMG  for IMG in os.listdir(args.input_dir)]          
    
    else:   
        
        folders = {args.input_dir + w: os.listdir(args.input_dir + w) for w in os.listdir(args.input_dir) if (len(w.split(".")) == 1) and ("@" not in w)}
    
        pictures = [k + "/" + v for k in folders.keys() if k in args.subfolders for v in folders[k]]
    
    
    r = re.compile(".*(" + args.prefix + ")")
    pictures = list(filter(r.match, pictures ))
    
    print("The number of piplines pictures {}".format(len(pictures)))
    
    DB = GenerateFaces(pictures, plot=False, thr=args.thr)

    if args.run:
        DB.run(archive=True)
    
    if args.cluster:
        labelIDs = FaceCluster(IDs = args.IDs).Cluster()
        FaceImageGenerator().GenerateImages(labelIDs, "ClusteredFaces", "Montage", IDs=args.IDs) 


if __name__ == '__main__':
    main()