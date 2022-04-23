#!/bin/bash


source1=${2:-empty}
source2=${3:-empty}
source3=${4:-empty}
config=${5:-secrets.cfg}
output=${6:-../Reolink/log/}
clf=${5:-FaceRecognition/KNN_weighted.json}

if [ $source1 = Eingang ]
then
  echo "Lets run smart surveillance on camera source: $source1"
  
  grep -A3 $source1 $config  | head -n4 | grep '^ip' > tmp33 && source tmp33 && rm tmp33
  grep -A3 $source1 $config  | head -n4 | grep '^username' > tmp33 && source tmp33 && rm tmp33
  grep -A3 $source1 $config  | head -n4 | grep '^password' > tmp33 && source tmp33 && rm tmp33

  python ../Reolink/motion_alert.py -s $source1 -c $config -o $output &

  container=$(docker ps)
  name=$(docker ps --format "{{.Names}}")
  echo $name
  echo $container

  docker exec $name pip install matplotlib
  docker exec $name pip install scikit-image
  sleep 3
  
  docker exec $name python3 app.py --input-uri rtsp://$username:$password@$ip:554/h264Preview_01_main --mot --txt WORKS.log --output-uri output/"$source1".mp4 &
  
    cp -R "$output"Reolink_motion_alerts_"$source1".log log/Reolink_motion_alerts_"$source1".log
  
#   ln -sf "{$output}Reolink_motion_alerts_{$source1}.log" "/home/brunosa3/projects/smart_surveillance/scr/smart_surveillance/log/Reolink_motion_alerts_{$source1}.log"

  sleep 5 
    python monitorLog.py -s $source1 -clf $clf &

fi

if [ $source2 = Terrasse ]
then
  echo "Lets run smart surveillance on camera source: $source2"
  
  grep -A3 $source2 $config  | head -n4 | grep '^ip' > tmp33 && source tmp33 && rm tmp33
  grep -A3 $source2 $config  | head -n4 | grep '^username' > tmp33 && source tmp33 && rm tmp33
  grep -A3 $source2 $config  | head -n4 | grep '^password' > tmp33 && source tmp33 && rm tmp33

  python ../Reolink/motion_alert.py -s $source2 -c $config -o $output &

  container=$(docker ps)
  name=$(docker ps --format "{{.Names}}")
  echo $name
  echo $container

  docker exec $name pip install matplotlib
  docker exec $name pip install scikit-image
  docker exec $name python3 app.py --input-uri rtsp://$username:$password@$ip:554/h264Preview_01_main --mot --txt WORKS.log --output-uri output/"$source2".mp4 &

  cp -R "$output"Reolink_motion_alerts_"$source2".log log/Reolink_motion_alerts_"$source2".log

#   ln -sf "{$output}Reolink_motion_alerts_{$source2}.log" "/home/brunosa3/projects/smart_surveillance/scr/smart_surveillance/log/Reolink_motion_alerts_{$source2}.log"

  sleep 5 
  python3 monitorLog.py -s $source2 -clf $clf &
fi

if [ $source3 = Carport ]
then
  echo "Lets run smart surveillance on camera source: $source3"
  
  grep -A3 $source3 $config  | head -n4 | grep '^ip' > tmp33 && source tmp33 && rm tmp33
  grep -A3 $source3 $config  | head -n4 | grep '^username' > tmp33 && source tmp33 && rm tmp33
  grep -A3 $source3 $config  | head -n4 | grep '^password' > tmp33 && source tmp33 && rm tmp33

  python ../Reolink/motion_alert.py -s $source3 -c $config -o $output &

  container=$(docker ps)
  name=$(docker ps --format "{{.Names}}")
  echo $name
  echo $container

  docker exec $name pip install matplotlib
  docker exec $name pip install scikit-image
  docker exec $name python3 app.py --input-uri rtsp://$username:$password@$ip:554/h264Preview_01_main --mot --txt WORKS.log --output-uri output/"$source3".mp4 &
  
  cp -R "$output"Reolink_motion_alerts_"$source3".log log/Reolink_motion_alerts_"$source3".log
#   ln -sf "{$output}Reolink_motion_alerts_{$source3}.log" "/home/brunosa3/projects/smart_surveillance/scr/smart_surveillance/log/Reolink_motion_alerts_{$source3}.log"

  sleep 5 
  python3 monitorLog.py -s $source3 -clf $clf &
fi

if [ $source1 = Video ]
then
  echo "Lets run smart surveillance on camera source: $source1"
  
  container=$(docker ps)
  name=$(docker ps --format "{{.Names}}")
  echo $name
  echo $container

  docker exec $name pip install matplotlib
  docker exec $name python3 app.py --input-uri $config --mot --txt WORKS.log --output-uri output/"$source1".mp4 &
  

  
#   ln -sf "{$output}Reolink_motion_alerts_{$source1}.log" "/home/brunosa3/projects/smart_surveillance/scr/smart_surveillance/log/Reolink_motion_alerts_{$source1}.log"

fi

#   sleep 5 
#   python3 monitorLog.py -s $source1 -clf $clf &
if [ $source1 != Eingang ] &&  [ $source1 != Video ] &&  [ $source2 != Terrasse ] &&  [ $source3 != Carport ]
then
  echo "you did not match any of the supported source options:
  
  -source1: Eingang | Video
  -source2: Terrasse 
  -source3: Carport
  
  your choise was:
  -source1: $source1
  -source2: $source2
  -source3: $source3"


else
  echo "your choise was:
  -source1: $source1
  -source2: $source2
  -source3: $source3
  -config: $config
  -output: $output
  -clf: $clf"
  
fi


# if $source
# grep -A3 $source $config  | head -n4 | grep '^ip' > tmp && source tmp && rm tmp
# grep -A3 $source $config  | head -n4 | grep '^username' > tmp && source tmp && rm tmp
# grep -A3 $source $config  | head -n4 | grep '^password' > tmp && source tmp && rm tmp

# python /home/brunosa3/projects/smart_surveillance/scr/Reolink/motion_alert.py -s $source -c $config -o $output &

# container=$(docker ps)
# name=$(docker ps --format "{{.Names}}")
# echo $name
# echo $container

# docker exec $name pip install matplotlib
# docker exec $name python3 app.py --input-uri rtsp://$username:$password@$ip:554/h264Preview_01_main --mot --txt WORKS.log &
# docker exec $name python3 app.py --input-uri videos/video-output-99B2EA0E-A6B6-48A5-ACFB-16E2CA051FEE.mov --mot --txt WORKS.log &


# sleep 5 
# python3 monitorLog.py -s $source -clf $clf &
