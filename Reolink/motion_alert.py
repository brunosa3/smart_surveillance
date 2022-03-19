from reolinkapi import Camera
import argparse
import logging
import os
import sys
from datetime import datetime, timedelta
from reolinkapi import Camera
from configparser import RawConfigParser
from pathlib import Path


parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
optional = parser._action_groups.pop()
required = parser.add_argument_group('required arguments')
required.add_argument('-s', '--source', required=True, help=
                          'source of Reolink camera\n'
                          '1) Eingang\n'
                          '2) Terrasse\n'
                          '3) Carport\n')
optional.add_argument('-c', '--config', metavar="FILE",
                          default='/home/brunosa3/secrets.cfg',
                          help='path to credetials')
optional.add_argument('-o', '--output',
                          default="./",
                          help='output log file')
parser._action_groups.append(optional)
args = parser.parse_args()

    


def read_config(props_path: str) -> dict:
    """Reads in a properties file into variables.
    
    this config file is kept out of commits with .gitignore. The structure of this file is such:
    """
    config = RawConfigParser()
    assert os.path.exists(props_path), f"Path does not exist: {props_path}"
    config.read(props_path)
    return config


# Get the local time on the camera
def get_time(cam: Camera) -> datetime:
    data = cam.get_dst()[0]['value']['Time']
    return datetime(data['year'], data['mon'], data['day'], data['hour'], data['min'], data['sec'])


# set up logging
logging.basicConfig(format='%(asctime)s [%(levelname)8s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S', 
                    filename= args.output + 'Reolink_'+ args.source +'.log', level=logging.INFO)

# Read in your ip, username, & password
config = read_config(args.config)

ip = config.get(args.source, 'ip')
un = config.get(args.source, 'username')
pw = config.get(args.source, 'password')

cam = Camera(ip, un, pw)

# must first login since I defer have deferred the login process
cam.login()

start = get_time(cam) 
alarm = cam._execute_command('GetMdState', [{ "cmd":"GetMdState",  "param":{ "channel":0}}] )[0]
performance = cam.get_performance()[0]
alarm_area = cam._execute_command('GetMdState', [{"cmd": "GetMdAlarm", "action": 1, "param": {"channel": 0}}])[0]
enc = cam._execute_command('GetEnc', [{"cmd": "GetEnc", "action":1,"param":{"channel":0}}])[0]

event = 0


logging.info(f"===================== Start Monitoring Motion of Reolink Camera ====================")
logging.info(f"{'camera source:':<25}{args.source}, {'frameRate:':<25}{enc['value']['Enc']['mainStream']['frameRate']}")
logging.info(f"{'resolution:':<25}{enc['value']['Enc']['mainStream']['size']}, {'bitRate:':<25}{enc['value']['Enc']['mainStream']['bitRate']}")
logging.info(f"------------------------------------------------------------------------------------")
logging.info(f"{'start time:':<25}{start}")
logging.info(f"====================================================================================")
logging.info(f"{'alarm area'} {alarm_area['value']['MdAlarm']['scope']['cols'], alarm_area['value']['MdAlarm']['scope']['rows']}{':'}{alarm_area['value']['MdAlarm']['scope']['table']}")
logging.info(f"====================================================================================")
def main():
    while True:    
        dst = get_time(cam) 
        alarm = cam._execute_command('GetMdState', [{ "cmd":"GetMdState",  "param":{ "channel":0}}] )[0]
        performance = cam.get_performance()[0]
        if alarm['value']['state'] == 1:
            logging.info(f"{dst}, {args.source}, {alarm['cmd']}, {alarm['code']}, {alarm['value']['state']}")
            global event 
            event += 1

    #     logging.info(f"{performance['cmd']}, {performance['code']}, {performance['value']['Performance']['codecRate']}, {performance['value']['Performance']['cpuUsed']}, {performance['value']['Performance']['netThroughput']}")

if __name__ == '__main__':
    try:
        main()    
    except KeyboardInterrupt:
        print('Interrupted')        
        try:
            stop = get_time(cam)
            logging.info(f"===================== Stop Monitoring Motion of Reolink Camera =====================")
            logging.info(f"{'camera source:':<25}{args.source}")
            logging.info(f"{'stop time:':<25}{stop}")
            logging.info(f"{'monitoring time:':<25}{stop-start}")
            logging.info(f"{'monitoring counts:':<25}{event}")
            logging.info(f"====================================================================================")
            sys.exit(0)
        except SystemExit:
            os._exit(0)