#!/bin/bash

GRAB_OBJS=("airplane" "camera" "cylindermedium" "flashlight" "headphones" "piggybank" "spheremedium" "toothbrush" "watch"
"alarmclock" "cubelarge" "cylindersmall" "flute" "knife" "pyramidlarge" "spheresmall" "toothpaste" "waterbottle" "apple"
"cubemedium" "doorknob" "fryingpan" "lightbulb" "pyramidmedium" "stamp" "toruslarge" "wineglass" "banana" 
"cubesmall" "duck" "gamecontroller" "mouse" "pyramidsmall" "stanfordbunny" "torusmedium" "binoculars" "cup" 
"elephant" "hammer" "mug" "scissors" "stapler" "torussmall" "bowl" "cylinderlarge" "eyeglasses" "hand" "phone" "spherelarge" "teapot" "train")

USERS=("s6" "s7" "s8" "s9" "s10")

for user in "${USERS[@]}"
do
   for obj in "${GRAB_OBJS[@]}"
   do
      bash ./tools/scripts/submit.sh cpu-$user-$obj cpu python ./datasets/scripts/process_grab_rewrite.py --filter="$user/$obj"
   done
done