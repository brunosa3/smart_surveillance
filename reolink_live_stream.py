import argparse
import cv2
import os 
from reolinkapi import Camera
from configparser import RawConfigParser
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description='arguments to control the pipline')

    parser.add_argument('--config_path', default="/home/brunosa3/secrets.cfg", help='the path to the config file for accessing the camera')
    parser.add_argument('--which_camera', default='Eingang', help='defining which camera to access')
    parser.add_argument('--asyn', default='blocking', choices=["blocking", "non_blocking"], help='definwes if live strea,m is asynchronus')
    args = parser.parse_args()

    return args
                        
class video_stream:
    # get all the parsed arguments
    args = parse_args()
    
    def __init__(self, config_path=args.config_path, which_cam=args.which_camera, asyn = args.asyn):        
        # Read in your ip, username, & password
        self.config = self.read_config(config_path)
        self.ip = self.config.get(which_cam, 'ip')
        self.un = self.config.get(which_cam, 'username')
        self.pw = self.config.get(which_cam, 'password')
        print("self."+ asyn + "()")
        eval("self."+ asyn + "()")
        
    def read_config(self, config_path: str) -> dict:
        """Reads in a properties file into variables.

        this config file is kept out of commits with .gitignore. The structure of this file is such:
        """
        config = RawConfigParser()
        assert os.path.exists(config_path), f"Path does not exist: {config_path}"
        config.read(config_path)
        return config
    
    
    def non_blocking(self):
#     print("calling non-blocking")
        def inner_callback(self, img):
            cv2.imshow("name", maintain_aspect_ratio_resize(img, width=600))
            print("got the image non-blocking")
            key = cv2.waitKey(1)
            if key == ord('q'):
                cv2.destroyAllWindows()
                exit(1)

        c = Camera(self.ip, self.un, self.pw)
        # t in this case is a thread
        t = c.open_video_stream(callback=inner_callback)

        print(t.is_alive())
        while True:
            if not t.is_alive():
                print("continuing")
                break
            # stop the stream
            # client.stop_stream()


    def blocking(self):
        c = Camera(self.ip, self.un, self.pw)
        # stream in this case is a generator returning an image (in mat format)
        stream = c.open_video_stream()

        # using next()
        # while True:
        #     img = next(stream)
        #     cv2.imshow("name", maintain_aspect_ratio_resize(img, width=600))
        #     print("got the image blocking")
        #     key = cv2.waitKey(1)
        #     if key == ord('q'):
        #         cv2.destroyAllWindows()
        #         exit(1)

        # or using a for loop
        for img in stream:
            print("img stream")
            cv2.imshow("name", self.maintain_aspect_ratio_resize(img, width=600))
            print("got the image blocking")
            key = cv2.waitKey(1)
            if key == ord('q'):
                cv2.destroyAllWindows()
                exit(1)

    # Resizes a image and maintains aspect ratio
    def maintain_aspect_ratio_resize(self, image, width=None, height=None, inter=cv2.INTER_AREA):
        # Grab the image size and initialize dimensions
        dim = None
        (h, w) = image.shape[:2]

        # Return original image if no need to resize
        if width is None and height is None:
            return image

        # We are resizing height if width is none
        if width is None:
            # Calculate the ratio of the height and construct the dimensions
            r = height / float(h)
            dim = (int(w * r), height)
        # We are resizing width if height is none
        else:
            # Calculate the ratio of the 0idth and construct the dimensions
            r = width / float(w)
            dim = (width, int(h * r))

        # Return the resized image
        return cv2.resize(image, dim, interpolation=inter)

    
video_stream()
