# Welcome to our capstone project "smart survaillance"
Welcome to our capstone project "smart survaillance" for our Master of Applied Data Science from the University of Michigan School of Information.

The authors Akshar and Sandro formed a team to reduce annoying alerts of our current survaillance system by applying computer vison to identify if a tracked person on the video stream is unkown.

A detailed report blog can be viewed at: [https://capstone.analyticsoverflow.com/report](https://capstone.analyticsoverflow.com/report)

## Data Challenge
One of the biggest challenges that we faced early in our project was that there was no dataset "in the wild" that we could use for implementing our project since we wanted to personalize our system to actual users' environments which meant that we would need to know the personalization features to apply. Therefore, all of the data that we would need for training, testing, and evaluation would have to come from our own devices and sources. Here, our team member, Sandro Bruno generously offered his own home and camera setup to be the source of our dataset and fulfill our data requirements.

The streaming video feeds use a set of off-the-shelf IP cameras (Reolink RLC-423) and are run 24x7. There are actually a set of 6 cameras capturing video streams but our project will focus on a single stream from the camera labeled Eingang (translated to entrance). The stream is stored in a local storage array that has the capacity to store a few weeks worth of data. 

## Compilation Challenge
As one can see on our building blocks of our current system below - it is a more complex and interdependent system which we accomplished on our local machines. 
![alt text](pics/building_blocks.png)
However, it was too hard for us in that short time to make sure tha all components work in other enviroments. Therefore, we have decided to keep that out of scope for this repository. However, in scope of this repository is to demonstrate smaller parts of our system. This together with our detailed report and our [analysis tool](https://capstone.analyticsoverflow.com/analysis) should help you to understand what we have archived. In near future we would like to expand this or another repository to also include the missing building blocks here such that you can also run our system in your enviroment with litle changes.

## Getting started
After you have cloned our repository you should recreate our conda enviroment such that you can reproduce our work
```python
conda env create -f smart_surveillance.yml
```
Next we need to compile the multi object tracker we have customized from [FastMOT](https://github.com/GeekAlexis/FastMOT)
### Install for x86 Ubuntu
Make sure to have [nvidia-docker](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker) installed. The image requires NVIDIA Driver version >= 450 for Ubuntu 18.04 and >= 465.19.01 for Ubuntu 20.04. We have built and test it stabely with Driver version 510.47.03 and CUDA Version 11.6 on Ubuntu 20.04:
```terminal
# change directory
cd smart_surveillance/
# build docker
docker build --build-arg TRT_IMAGE_VERSION=21.05 -t  fastmot:latest .
# for displaying within docker
xhost local:root
```
Now we can check if the driver was installed correctly 
```terminal
# check driver 
sudo docker run --gpus all nvidia/cuda:11.0-base nvidia-smi
```

Now we can compile the docker by going in the *plugin* folder and compile it using make
```
cd fastmod/plugins
make
```
then you need to go in the *scripts/* folder and run *download_models.sh* to fetch all the model files
```terminal
cd scripts/
./download_models.sh
```

Now we should be able to launch the docker in an interactive mode - Lets test it via
```docker
# run/open docker
docker run --gpus all --rm -it -v $(pwd):/usr/src/app/FastMOT -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=unix$DISPLAY -e TZ=$(cat /etc/timezone) fastmot:latest
```

#### Optional
check if you have cv2 available within the launched interactive docker if not you need to install cv2 via wheel within the docker to get all dependencies otherwise it will not work becuase GSTREAMER support for videocapture is not given otherwise
``` docker
pip install --no-binary opencv-python opencv-python
```
try if you can run the tracker on a test video just copy paste any video of your choice (with some people walking around) into the docker folder
```docker
python3 app.py --input-uri fastmod/REC_84194.mov --mot -o test.mp4
```
in case you had to manually install a dependency within the docker e.g. cv2 rember the id of the container indicated in the prompt *root@containerID* and leave the interactive docker via the command *exit*  

In order to display a list of launched containers including yours you need to execute
```terminal
sudo docker ps -a
```
Finally, create a new image by committing the changes using the following syntax
``` terminal
# sudo docker commit [CONTAINER_ID] [new_image_name]
sudo docker commit fa9f6b05cde3 smart_surveillance
# now you should see the saved container in docker images
docker images
```
Now we can launch the saved container
```
docker run --gpus all --rm -it -v $(pwd):/usr/src/app/FastMOT -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=unix$DISPLAY -e TZ=$(cat /etc/timezone) smart_surveillance:latest
# or
docker run --gpus all --rm -it -v $(pwd):/usr/src/app/FastMOT -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=unix$DISPLAY -e TZ=$(cat /etc/timezone) containerID_seen_in_docker_image
```

### using the system with your surveillance camera
Any Reolink camera that has a web UI should be compartible with our system eg.: 
+ RLC-411WS
+ RLC-423 (validated on our own cameras)
+ RLC-420-5MP
+ RLC-410-5MP
+ RLC-520
+ C1-Pro
+ D400

you just need to create a *secrets.cfg* file in the *smart_surveillance* directory that contains the credentials for each Reolink camera you want to access
```
# sudo nano smart_surveillance/secrets.cfg
[camera1]
ip=xxx.xxx.xxx.xxx
username=mustermann
password=12345
[camera2]
ip=xxx.xxx.xxx.xxx
username=mustermann
password=12345
[camera3]
ip=xxx.xxx.xxx.xxx
username=mustermann
password=12345
[camera4]
ip=xxx.xxx.xxx.xxx
username=mustermann
password=12345
[camera5]
ip=xxx.xxx.xxx.xxx
username=mustermann
password=12345
```

Now you should be able to access your Reolink camera with its customized alert areas and monitor the incoming alerts from your Reolink camera 
```terminal
# $source need to be replaced by the camera name setted in secrets.cfg e.g. camera1 
python Reolink/motion_alert.py -s $source -c smart_surveillance/secrets.cfg -o Reolink/log/ &
```

## Demonstrations
The notebooks mentioned below should demonstrate the following parts
### 1) overcome a cold start of our system [01_Overcome_cold_start_face_recognition.ipynb](01_Overcome_cold_start_face_recognition.ipynb)
#### quick start
configure the paths for the different files on your machine in the configuration file [conf.json](conf.json)  
*NOTE: the weights and architecture of FaceNet are stored in [keras-facenet-h5](keras-facenet-h5)*

To detect, extract and cluster all the detected faces of your personal picture collection you need to run the script below in the terminal. This will output folders of faces that were clustered together and seem to be the identical person. In addition it will store all the extracted encodings, as well as paths to the cropped faces and original pictures in a json file.

```terminal
python main.py -c conf.json -e True -cl True
```
Next you need to manually evaluate the correctnes of the clusters and label the folders with the appropiate name of the subjects. Once this was done you can run the code below to restructure the created database. This database can be used now to reidentify these people.

```terminal
python main.py -c conf.json -r True
```
### 2) comparison of incoming alerts between the IP camera and YOLO/DeepSORT [02_evaluation_of_alerting_system.ipynb](02_evaluation_of_alerting_system.ipynb)

### 3) how to detect faces 

### 4) how to upsample low resolution images

### 5) how we have decided in or out of interest area?

### 4) evaluation of our models [03_model_evaluation.ipynb](03_model_evaluation.ipynb)   
*NOTE: the extensive manually labeled ground truth dataset is stored in [groundTruth](groundTruth) and the precomputed evaluation is stored in [FaceRecognition_evaluation.json](FaceRecognition_evaluation.json)*

