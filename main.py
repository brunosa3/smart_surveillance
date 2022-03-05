import argparse
import os
import re
from FaceEncoder import *
import json
import warnings
warnings.filterwarnings('ignore')

def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError(f'{value} is not a valid boolean value')

def parse_args():
    parser = argparse.ArgumentParser(description='arguments to control the pipline')
    
    # construct the argument parser and parse the arguments

    parser.add_argument("-c", "--conf", required=True, help="path to the JSON configuration file")
    parser.add_argument("-e", "--run", type=str_to_bool, nargs='?', const=True, default=False, help="do you want to encode")
    parser.add_argument("-cl", "--cluster",type=str_to_bool, nargs='?', const=True, default=False, help="do you want to cluster")
    parser.add_argument("-r", "--restructureDB", type=str_to_bool, nargs='?', const=True, default=False, help="do you want to restructure")
    args = vars(parser.parse_args())

    conf = json.load(open(args["conf"]))

    conf.update({k:v for k,v in args.items() if k != "conf"})

    # client    
    return conf



# that is the pipline backbone - deciding what steps to trigger
def main():
    # get all the parsed arguments
    conf = parse_args()
    
    # check if we need to consider all sub-folders within specified directory    
    if conf["subfolders"] == "all":
        # walk through each subfolder till you reach a filename in list comprehension style
        pictures = [os.path.join(root, name)
        for root, dirs, files in os.walk(conf["input_dir"])
            for name in files
                if name.endswith((tuple(i for i in conf["prefix"].split("|"))))]
        
        # for me i found files that included @ signs which I had to filter out - don t know if thats true for you too
        # seemed like synology sys files that are created for shared folders
        pictures = [i for i in pictures if "@" not in i]
        
    # only use the pictures which are in the specified folder (not sub-folders)    
    elif conf["subfolders"] == None:
        
        # extract all the paths to the images in the specified folder
        pictures = [conf["input_dir"] + IMG  for IMG in os.listdir(conf["input_dir"])]          
    
    # look in specified sub-folders
    else:   
        
        # identify all the subfolders within the specified directory
        # this could be done more elegant with the os.path.isdir() and os.access(file_path, os.R_OK) functions
        folders = {conf["input_dir"] + w: os.listdir(conf["input_dir"] + w) for w in os.listdir(conf["input_dir"]) if (len(w.split(".")) == 1) and ("@" not in w)}
        
        # extract all the paths to the pictures within the specified subfolders
        pictures = [k + "/" + v for k in folders.keys() if k in conf["subfolders"] for v in folders[k]]
    
    # regex matching all pictures with the specified prefix-ignoring everything else
    r = re.compile(".*(" + conf["prefix"] + ")")
    pictures = list(filter(r.match, pictures ))
    
    print("[INFO] preparing pipline")
    
    
    # initiate the pipline with the specified arguments
    DB = GenerateFaces(file_path=pictures, output=conf["output_dir"], plot=conf["plot"], thr=conf["thr"], model_path=conf["model_path"], weight_path=conf["weight_path"], database=conf["DB_path"])
    
    # decide if we need to run the face extraction and encoding pipline
    if conf["run"]:
        print("[INFO] starting to extract faces faces from {} pictures".format(len(pictures)))
        print("[INFO] specified input directory {}, output_directory {}, plot {}, confidence threshold {}".format(conf["input_dir"], conf["output_dir"], conf["plot"], conf["thr"]))
        print("[INFO] you want to archive the extractions {}, batch_size {}".format(conf["archive"], conf["batch_size"]))
        DB.run(archive=conf["archive"], batch_size=conf["batch_size"])
    
    # decide if we need to run the clustering pipline
    if conf["cluster"]:
        print("[INFO] preparing clustering of extracted faces")
        print("[INFO] specified database directory {}, specified IDs {}".format(conf["DB_path"], conf["IDs"]))
        
        # cluster the extracted faces  
        labelIDs = FaceCluster(IDs = conf["IDs"], file_path=conf["DB_path"]).Cluster()  
        
        # output the clusterings in folders of specified folder names 
        FaceImageGenerator().GenerateImages(labels=labelIDs, file_path= conf["DB_path"], OutputFolderName =conf["cluster_folder_name"], MontageOutputFolder = conf["montage_folder_name"], IDs=conf["IDs"]) 
        
    if conf["restructureDB"]:
        print("[INFO] start to restructure the database")
        new_DB = DB.restructure_DB(cluster_path= conf["cluster_path"],
                          new_DB_path = conf["new_DB_path"],
                          ignore= conf["ignore"],
                          ig_faces= conf["ig_faces"])
        
        print("[INFO] There are {} subjects in the reference database".format(len(new_DB)))


if __name__ == '__main__':
    main()