import time
import re
from FaceRecognition_decoupled import *
import joblib    
import time as TIME
import logging
import argparse
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

startTime = datetime.now()

# # set up logging
# logging.basicConfig(format='%(asctime)s [%(levelname)8s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S', 
#                     filename= 'log/test_FaceNet.log', level=logging.INFO)

formatter = logging.Formatter('%(asctime)s [%(levelname)8s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')


def setup_logger(name, log_file, level=logging.INFO):
    """To setup as many loggers as you want"""

    handler = logging.FileHandler(log_file)        
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger

knn_logger = setup_logger('KNN_logger', 'log/KNN_new_FaceRecognition.log')
SGDC_logger = setup_logger('SGDC_logger', 'log/SGDC_new_FaceRecognition.log')

# knn_logger = setup_logger('KNN_logger', 'log/KNN_test.log')
# SGDC_logger = setup_logger('SGDC_logger', 'log/SGDC_test.log')



logfile = open('log/YOLO4_DeepSORT_Eingang_overview.log',"r")
snap = {"date": 1,
        "time": 2,
        "log": 3,
        "event": 4,
        "filename": 5,
        "source": 6,
        "frame": 7,
        "trkID": 8,
        "x": 9,
        "y": 10,
        "w": 11,
        "h": 12
       }
knn = joblib.load("FaceRecognition/KNN_weighted.json")
SGDC = joblib.load("FaceRecognition/SGDC.json")
fn = import_FaceNet()

ids = np.zeros(10000)
knn_IDs = {}
SGDC_IDs = {}
ID = {}

p = re.compile("(\d+-\d+-\d+)\s*(\d+:\d+:\d+)\s*\[\s*(\w+)\]\s*([\w_]+):\s*([-\d _\w/.]+),\s*(\w+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+\.\d+)\s*,\s*(-?\d+\.\d+)\s*,\s*(-?\d+\.\d+)\s*,\s*(-?\d+\.\d+)\s*")
n = 0
for line in tqdm(logfile):
    print(line)
    try:
        if re.match(".*(Found|Lost|Reidentified|Out).*", line):
    #         print(line)
            log = re.compile("(\d+-\d+-\d+)\s*(\d+:\d+:\d+)\s*\[\s*\w+\]\s*(\w+)\s*:\s*(\w+)\s*(-?\d+)\s*\w+\s*\(\s*(-?\d+\s*,\s*-?\d+\s*)\)")
            try:
                date = log.search(line).group(1)
                time = log.search(line).group(2)
                event = log.search(line).group(3)
                oj = log.search(line).group(4)
                trkID = log.search(line).group(5)
                cord = log.search(line).group(6)
                cont = f"{date}, {time}, {event}, {oj}, {trkID}, {cord}"

            except Exception as e:
                print(e)
                print("error occured in .*(Found|Lost|Reidentified|Out).* snipped")
                print(line)
                cont1 = e


            if (event == "Found") | (event == "Reidentified"):
                ids[int(trkID)] += 1
                T = datetime.strptime(time, '%H:%M:%S')
                
                knn_NEW_PATH = "FaceRecognition/on_queue/KNN/"+trkID+"/"+date+"/"+str(T.hour)+"/" 
                SGDC_NEW_PATH = "FaceRecognition/on_queue/SGDC/"+trkID+"/"+date+"/"+str(T.hour)+"/"

#                 knn_NEW_PATH = "on_queue/KNN/"+trkID+"/"+date+"/"+str(T.hour)+"/" 
#                 SGDC_NEW_PATH = "on_queue/SGDC/"+trkID+"/"+date+"/"+str(T.hour)+"/"
                Path(knn_NEW_PATH).mkdir(parents=True, exist_ok=True)
                Path(SGDC_NEW_PATH).mkdir(parents=True, exist_ok=True)

                knn_IDs[str(trkID)] = {"Path": knn_NEW_PATH + '_'.join([date, time.replace(":", "_"), trkID]) + '.log',
                      "events": [cont]}
                SGDC_IDs[str(trkID)] = {"Path": SGDC_NEW_PATH + '_'.join([date, time.replace(":", "_"), trkID]) + '.log',
                      "events": [cont]}
            else:
                ids[int(trkID)] = 0
                knn_IDs[str(trkID)]["events"] += [cont]
                SGDC_IDs[str(trkID)]["events"] += [cont]
                myfile = open(knn_IDs[str(trkID)]["Path"], 'w')
                for L in knn_IDs[str(trkID)]["events"]:
                    myfile.write(L)
                    myfile.write('\n')
                myfile.close()     
                myfile = open(SGDC_IDs[str(trkID)]["Path"], 'w')
                for L in SGDC_IDs[str(trkID)]["events"]:
                    myfile.write(L)
                    myfile.write('\n')
                myfile.close()     
            knn_logger.info(cont)
            SGDC_logger.info(cont)

        elif re.match(".*Snap.*", line):  
            trkID = p.search(line).group(snap["trkID"])

            if (ids[int(trkID)] > 0):
    #             print("that the line parsed to FaceNet", line)
                dist, ident, enc, SGDC_dist, SGDC_ident = FaceRecognition().who_is_it(image_path= p.search(line).group(snap["filename"]), 
                                            thr=0.75, plot=False, FaceNet=fn, knn=knn,SGDC=SGDC, res_model="espcn", zoom=4)                
                try:                    
    #                 print(dist)
    #                 print(ident)

                    dat = p.search(line).group(snap["date"])
                    tim = p.search(line).group(snap["time"]) 
                    src = p.search(line).group(snap["source"])
                    frm = p.search(line).group(snap["frame"])
                    trk = p.search(line).group(snap["trkID"])
                except Exception as e:
                    print(e)
                    print("error occured in ..*Snap.* snipped")

                ### knn if unknown or face not visible
                if dist is None:
                    if ident[0] == "unknown":
                        cont1 = f"{dat}, {tim}, {src}, {frm}, {trk}, {ident[0]}, {'NA'}, {' '.join([str(en) for en in enc.tolist()])}"
                        knn_logger.info(cont1)                   
                        ids[int(trkID)] += 1 
                        try:
                            if p.search(line).group(snap["trkID"]) in list(knn_IDs.keys()):
                                knn_IDs[p.search(line).group(snap["trkID"])]["events"] += [cont1] #[(ident[0], dist, enc.tolist())]

                        except Exception as e:
                            print(e)
                            print("KNN error occured in ..*Snap.* unknown person")  
                    else:
                        cont1 = f"{dat}, {tim}, {src}, {frm}, {trk}, {ident[0]}, {'NA'}, {'-1'}"
                        knn_logger.info(cont1)
                        if p.search(line).group(snap["trkID"]) in list(knn_IDs.keys()):
                            knn_IDs[p.search(line).group(snap["trkID"])]["events"] += [cont1] #[(ident[0], 'NA', -1)]


                else:                                           
                    ids[int(trkID)] += 1 
                    try:
                        cont1 = f"{dat}, {tim}, {src}, {frm}, {trk}, {ident[0]}, {str(dist)}, {' '.join([str(en) for en in enc.tolist()])}"
                        if p.search(line).group(snap["trkID"]) in list(knn_IDs.keys()):
                            knn_IDs[p.search(line).group(snap["trkID"])]["events"] += [cont1] #[(ident[0], dist, enc.tolist())]

                        knn_logger.info(cont1)
                    except Exception as e:
                        print(e)
                        print("KNN error occured in ..*Snap.* when object was lost snipped")

                ### SGDC if unknown
                if SGDC_dist is None:
                    if SGDC_ident[0] == "unknown":
                        cont2 = f"{dat}, {tim}, {src}, {frm}, {trk}, {SGDC_ident[0]}, {'NA'}, {' '.join([str(en) for en in enc.tolist()])}"
                        SGDC_logger.info(cont2)                   
                        ids[int(trkID)] += 1 
                        try:
                            if p.search(line).group(snap["trkID"]) in list(SGDC_IDs.keys()):
                                SGDC_IDs[p.search(line).group(snap["trkID"])]["events"] += [cont2] #[(ident[0], dist, enc.tolist())]

                        except Exception as e:
                            print(e)
                            print("SGDC error occured in ..*Snap.* unknown person")  

                    else:
                        cont2 = f"{dat}, {tim}, {src}, {frm}, {trk}, {SGDC_ident[0]}, {'NA'}, {'-1'}"
                        SGDC_logger.info(cont2)
                        if p.search(line).group(snap["trkID"]) in list(SGDC_IDs.keys()):
                            SGDC_IDs[p.search(line).group(snap["trkID"])]["events"] += [cont2]


                else:                                           
                    ids[int(trkID)] += 1 
                    try:
                        cont2 = f"{dat}, {tim}, {src}, {frm}, {trk}, {SGDC_ident[0]}, {str(SGDC_dist)}, {' '.join([str(en) for en in enc.tolist()])}"
                        if p.search(line).group(snap["trkID"]) in list(SGDC_IDs.keys()):
                            SGDC_IDs[p.search(line).group(snap["trkID"])]["events"] += [cont2] #[(ident[0], dist, enc.tolist())]

                        SGDC_logger.info(cont2)
                    except Exception as e:
                        print(e)
                        print("SGDC error occured in ..*Snap.* when object was lost snipped")

            else:
                print("the FaceRecognition is not turned off nor on mode equal {}".format(ids[int(trkID)]))
    #     if n == 40:
    #         break
    #     n += 1
    except Exception as e:
        print(e)
        continue
    
print(datetime.now() - startTime)