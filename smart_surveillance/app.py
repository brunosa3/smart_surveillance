#!/usr/bin/env python3

from pathlib import Path
from types import SimpleNamespace
import argparse
import logging
import json
import cv2

import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import re
from configparser import RawConfigParser
import os

import fastmot
import fastmot.models
from fastmot.utils import ConfigDecoder, Profiler

from skimage.transform import resize
with open("log/Reolink_motion_alerts_Eingang.log", "r") as f: 
    lines = f.readlines()
for l in lines:
    if re.match(".*resolution.*",l):
        pat = re.match(".*resolution:\s+(\d+)\*(\d+).*",l)
        w = int(pat.group(1))
        h = int(pat.group(2))
        print("[INFO] resolution of this camera is {}x{}".format(w,h))
    elif re.match(".*alarm area.*",l):
        pat = re.match(".*alarm.area.\((\d+),.(\d+)\):(\d+)",l)
        rows = int(pat.group(1))
        cols = int(pat.group(2))
        mask = pat.group(3)

        mask = resize(np.array([int(s) for s in mask]).reshape(cols,rows), (h,w))
        
        print(mask.shape)
        break


def in_or_out(mask, x1,y1,w,h, thr=80):
    x2, y2 = (int(x1+w), int(y1+h)) 
    rec = np.zeros(mask.shape)
    rec[y1:y2,x1:x2] = 100
    mask *= (255.0/mask.max())
    test = (mask-rec)
    in_out = test[int(y2*0.9):y2,x1:x2].mean() > thr
    return in_out

def read_config(props_path: str) -> dict:
    """Reads in a properties file into variables.
    """
    config = RawConfigParser()
    assert os.path.exists(props_path), f"Path does not exist: {props_path}"
    config.read(props_path)
    return config



def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    optional = parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    group = parser.add_mutually_exclusive_group()
    required.add_argument('-i', '--input-uri', metavar="URI", required=True, help=
                          'URI to input stream\n'
                          '1) image sequence (e.g. %%06d.jpg)\n'
                          '2) video file (e.g. file.mp4)\n'
                          '3) MIPI CSI camera (e.g. csi://0)\n'
                          '4) USB camera (e.g. /dev/video0)\n'
                          '5) RTSP stream (e.g. rtsp://<user>:<password>@<ip>:<port>/<path>)\n'
                          '6) HTTP stream (e.g. http://<user>:<password>@<ip>:<port>/<path>)\n')
    optional.add_argument('-c', '--config', metavar="FILE",
                          default=Path(__file__).parent / 'cfg' / 'mot.json',
                          help='path to JSON configuration file')
    optional.add_argument('-l', '--labels', metavar="FILE",
                          help='path to label names (e.g. coco.names)')
    optional.add_argument('-o', '--output-uri', metavar="URI",
                          help='URI to output video file')
    optional.add_argument('-t', '--txt', metavar="FILE",
                          help='path to output MOT Challenge format results (e.g. MOT20-01.txt)')
    optional.add_argument('-m', '--mot', action='store_true', help='run multiple object tracker')
    optional.add_argument('-s', '--show', action='store_true', help='show visualizations')
    group.add_argument('-q', '--quiet', action='store_true', help='reduce output verbosity')
    group.add_argument('-v', '--verbose', action='store_true', help='increase output verbosity')
    optional.add_argument('-sn', '--snap_n', default=20, type=int, help='number of frames after which we take a snap for FaceNet')
    optional.add_argument('-mp', '--max_p', default=5, type=int, help='max number of pictures for FaceNet')
    parser._action_groups.append(optional)
    args = parser.parse_args()
    if args.txt is not None and not args.mot:
        raise parser.error('argument -t/--txt: not allowed without argument -m/--mot')
    
    try:
        # Read in your ip
        config = read_config("secrets.cfg")
        c = [l for l in list(config.keys()) if l != "DEFAULT"]    
        ip_set = re.findall( r'[0-9]+(?:\.[0-9]+){3}', args.input_uri )[0]
        source = [C for C in c if config.get(C, "ip") == ip_set][0] 
    except Exception as e:
        print(e)
        source = "Video"
    print(source)

    new_path = "FaceRecognition/FaceNet_input/"+ source + "/"
    Path(new_path).mkdir(parents=True, exist_ok=True)
    date_time = datetime.now()
    filename = new_path + date_time.strftime('%Y%B%d_%H_%M_%S') + "_" + source + "alarm_area.png"
    plt.imsave(filename, np.float32(mask))
    
    # set up logging
    logging.basicConfig(format='%(asctime)s [%(levelname)8s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S', filename= 'log/YOLO4_DeepSORT_' + source + '_overview.log')
    logger = logging.getLogger(fastmot.__name__)
    if args.quiet:
        logger.setLevel(logging.WARNING)
    elif args.verbose:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    # load config file
    with open(args.config) as cfg_file:
        config = json.load(cfg_file, cls=ConfigDecoder, object_hook=lambda d: SimpleNamespace(**d))

    # load labels if given
    if args.labels is not None:
        with open(args.labels) as label_file:
            label_map = label_file.read().splitlines()
            fastmot.models.set_label_map(label_map)
    
    prev_tr_id = np.zeros(30000, dtype = np.int16)

    print(config.resize_to, args.input_uri, args.output_uri)
    print(config.stream_cfg)
    stream = fastmot.VideoIO(config.resize_to, args.input_uri, args.output_uri, **vars(config.stream_cfg))

    mot = None
    txt = None
    if args.mot:
        draw = args.show or args.output_uri is not None
        mot = fastmot.MOT(config.resize_to, **vars(config.mot_cfg), draw=draw)
        mot.reset(stream.cap_dt)
    if args.txt is not None:
        Path(args.txt).parent.mkdir(parents=True, exist_ok=True)
        txt = open(args.txt, 'w')
    if args.show:
        cv2.namedWindow('Video', cv2.WINDOW_AUTOSIZE)

    logger.info('Starting video capture...')
    stream.start_capture()
    try:
        with Profiler('app') as prof:
            while not args.show or cv2.getWindowProperty('Video', 0) >= 0:
                frame = stream.read()
                if frame is None:
                    break

                if args.mot:
                    mot.step(frame)

                    if txt is not None:
                        for track in mot.visible_tracks():
                            tl = track.tlbr[:2] / config.resize_to * stream.resolution
                            br = track.tlbr[2:] / config.resize_to * stream.resolution
                            w, h = br - tl + 1
                            
                            date_time = datetime.now()
                        
                            
                            check = prev_tr_id[track.trk_id]
                            if (check % args.snap_n) == 0:
                                print("crop for FaceNet track {}".format(track.trk_id))
                                
                                                                
                                w_org, h_org = track.tlbr[2:] - track.tlbr[:2] + 1
                                
                                if in_or_out(mask, x1=int(track.tlbr[0]), y1=int(track.tlbr[1]), w=int(w_org),h=int(h_org), thr=80):
                                
                                    crop = frame[int(track.tlbr[1]):int(track.tlbr[1]+h_org), int(track.tlbr[0]):int(track.tlbr[0]+w_org)]
                                    #filename = "FaceNet_input/" + str(int(check/args.snap_n)) + "_crop_track_" + str(track.trk_id) + ".png"
                                    new_path = "FaceRecognition/FaceNet_input/"+ source + "/in_of_interest_area/trackID_" + str(track.trk_id) + "/"
                                    Path(new_path).mkdir(parents=True, exist_ok=True)



                                    filename = new_path + date_time.strftime('%Y%B%d_%H_%M_%S') + "_" + source + str(int(mot.frame_count)) + "_crop_track_" + str(track.trk_id) + ".png"
                                    
                                    try:
                                        plt.imsave(filename, cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                                                                            
                                        if check > args.max_p*args.snap_n:
                                            prev_tr_id[track.trk_id] = 0
                                        else:
                                            prev_tr_id[track.trk_id] += 1

                                        logger.info(f"{'Snap:':<14}{filename},{source},{mot.frame_count},{track.trk_id},{tl[0]:.6f},{tl[1]:.6f},{w:.6f},{h:.6f}")
                                    
                                    except Exception as e:
                                        print(e)
                                        print("However, we continue running!")
                                        continue
                                    



                                else:
                                    print("detected person but not on area of interest!")
                                    logger.info(f"{'out_of_area:':<14}{-1},{source},{mot.frame_count},{track.trk_id},{tl[0]:.6f},{tl[1]:.6f},{w:.6f},{h:.6f}")
                                    crop = frame[int(track.tlbr[1]):int(track.tlbr[1]+h_org), int(track.tlbr[0]):int(track.tlbr[0]+w_org)]
                                    #filename = "FaceNet_input/" + str(int(check/args.snap_n)) + "_crop_track_" + str(track.trk_id) + ".png"
                                    new_path = "FaceRecognition/FaceNet_input/"+ source + "/out_of_interest_area/trackID_" + str(track.trk_id) + "/"
                                    Path(new_path).mkdir(parents=True, exist_ok=True)



                                    filename = new_path + date_time.strftime('%Y%B%d_%H_%M_%S') + "_" + source + str(int(mot.frame_count)) + "_crop_track_" + str(track.trk_id) + ".png"
                                    try:
                                        plt.imsave(filename, cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                                    except Exception as e:
                                        print(e)
                                        print("However, we continue running!")
                                        continue
                                
                            else:
                                prev_tr_id[track.trk_id] += 1
                                
                            
                                         
#                             txt.write(f'{date_time}, {source},{mot.frame_count},{track.trk_id},{tl[0]:.6f},{tl[1]:.6f},'
#                                       f'{w:.6f},{h:.6f},-1\n')

                if args.show:
                    cv2.imshow('Video', frame)
                    if cv2.waitKey(1) & 0xFF == 27:
                        break
                if args.output_uri is not None:
                    stream.write(frame)
    finally:
        # clean up resources
        if txt is not None:
            txt.close()
        stream.release()
        cv2.destroyAllWindows()

    # timing statistics
    if args.mot:
        avg_fps = round(mot.frame_count / prof.duration)
        logger.info('Average FPS: %d', avg_fps)
        mot.print_timing_info()


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        import sys
        import os
        print("Oops, something went wrong: {}".format(e), file=sys.stderr)
        import traceback
        top = traceback.extract_tb(sys.exc_info()[2])[-1]
        print('{} : {} in {} at line {}'.format(type(e).__name__, str(e), os.path.basename(top[0]), str(top[1])))
        logger.info(f"{'error:':<14}{'script stoped'}, {str(e)}")
        sys.exit(1)
         

