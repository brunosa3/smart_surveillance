import time
import re
from FaceRecognition import *
import joblib    
import time as TIME
import logging
import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
optional = parser._action_groups.pop()
required = parser.add_argument_group('required arguments')
required.add_argument('-s', '--source', required=True, help=
                          'source of Reolink camera\n'
                          '1) Eingang\n'
                          '2) Terrasse\n'
                          '3) Carport\n')
optional.add_argument('-clf', '--model',
                          default="FaceRecognition/KNN_weighted.json",
                          help='model to recognize faces')
parser._action_groups.append(optional)
args = parser.parse_args()


# set up logging
logging.basicConfig(format='%(asctime)s [%(levelname)8s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S', 
                    filename= 'log/FaceNet_' + args.source + '_overview.log', level=logging.INFO)

clf = joblib.load(args.model)
fn = import_FaceNet()

ids = np.zeros(10000)
IDs = {}


def follow(thefile):
    thefile.seek(0,2)
    while True:
        line = thefile.readline()
        if not line:
            TIME.sleep(0.1)
            continue
        yield line

if __name__ == '__main__':
    try:
        logfile = open('log/YOLO4_DeepSORT_' + args.source + '_overview.log',"r")
        loglines = follow(logfile)
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
        p = re.compile("(\d+-\d+-\d+)\s*(\d+:\d+:\d+)\s*\[\s*(\w+)\]\s*(\w+):\s*([\w+/]+.png),(\w+),(\d+),(\d+),(\d+\.\d+),(\d+\.\d+),(\d+\.\d+),(\d+\.\d+)")
        for line in loglines:
            if re.match(".*(Found|Lost|Reidentified).*", line):
                print(line)
                log = re.compile("(\d+-\d+-\d+)\s*(\d+:\d+:\d+)\s*\[\s*\w+\]\s*(\w+):\s*(\w+)\s*(\d+)\s*\w+\s*\(\s*(\d+,\s*\d+)\)")
                try:
                    date = log.search(line).group(1)
                    time = log.search(line).group(2)
                    event = log.search(line).group(3)
                    oj = log.search(line).group(4)
                    trkID = log.search(line).group(5)
                    cord = log.search(line).group(6)
                except Exception as e:
                    print(e)
                    print("error occured in .*(Found|Lost|Reidentified).* snipped")

                if (event == "Found") | (event == "Reidentified"):
                    ids[int(trkID)] += 1
                else:
                    ids[int(trkID)] = 0

                logging.info(f"{date}, {time}, {event}, {oj}, {trkID}, {cord}")

            elif re.match(".*Snap.*", line):  
                trkID = p.search(line).group(snap["trkID"])

                if ids[int(trkID)] == 1:
                    print("that the line parsed to FaceNet", line)
                    dist, ident, enc = FaceRecognition().who_is_it(image_path= p.search(line).group(snap["filename"]), 
                                                thr=0.75, plot=False, FaceNet=fn, clf=clf, res_model="espcn", zoom=4)                
                    try:                    
                        print(dist)
                        print(ident)

                        dat = p.search(line).group(snap["date"])
                        tim = p.search(line).group(snap["time"]) 
                        src = p.search(line).group(snap["source"])
                        frm = p.search(line).group(snap["frame"])
                        trk = p.search(line).group(snap["trkID"])
                    except Exception as e:
                        print(e)
                        print("error occured in ..*Snap.* snipped")


                    if dist is None:
    #                     txt =" ".join([dat, tim, src, frm, trk, ident[0], "NA"])
                        logging.info(f"{dat}, {tim}, {src}, {frm}, {trk}, {ident[0]}, {'NA'}, {'-1'}")
                    else:

                        ids[int(trkID)] = 0

    #                     txt =" ".join([dat, tim, src, frm, trk, ident[0], str(dist[0])])
                        try:
                            IDs[ p.search(line).group(snap["trkID"])] = ident[0]
                            logging.info(f"{dat}, {tim}, {src}, {frm}, {trk}, {ident[0]}, {str(dist)}, {' '.join([str(en) for en in enc.tolist()])}")
                        except Exception as e:
                            print(e)
                            print("error occured in ..*Snap.* when object was lost snipped")
                elif ids[int(trkID)] == 0:
                    try:
                        dat = p.search(line).group(snap["date"])
                        tim = p.search(line).group(snap["time"]) 
                        src = p.search(line).group(snap["source"])
                        frm = p.search(line).group(snap["frame"])
                        trk = p.search(line).group(snap["trkID"])
                    except Exception as e:
                        print(e)
                        print("error occured when object was lost snipped")

    #                 txt =" ".join([dat, tim, src, frm, trk, "No need to reidentify - identified as {}".format(IDs[p.search(line).group(snap["trkID"])]), "NA"])
                    try:
                        logging.info(f"{dat}, {tim}, {src}, {frm}, {trk}, {'No need to reidentify - identified as {}'.format(IDs[trk])}, {'NA'}, {'-1'}")
                    except Exception as e:
                        print(e)

                else:
                    print("the FaceRecognition is not turned off nor on mode equal {}".format(ids[int(trkID)]))



    # p.search(log).group(snap["event"])
    #             print(txt)
    
    except Exception as e:
        import sys
        print("Oops, something went wrong: {}".format(e), file=sys.stderr)
        import traceback
        top = traceback.extract_tb(sys.exc_info()[2])[-1]
        print('{} : {} in {} at line {}'.format(type(e).__name__, str(e), os.path.basename(top[0]), str(top[1])))
        logging.info(f"{'error:':<14}{'script stoped'}, {str(e)}")
        sys.exit(1)