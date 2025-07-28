#/bin/bash
# conda activate hand
python grab/graspxl_preprocessing.py --split 0 > ../data/graspXL/output0.log 2>&1 &
python grab/graspxl_preprocessing.py --split 1 > ../data/graspXL/output1.log 2>&1 &
python grab/graspxl_preprocessing.py --split 2 > ../data/graspXL/output2.log 2>&1 &
python grab/graspxl_preprocessing.py --split 3 > ../data/graspXL/output3.log 2>&1 &
python grab/graspxl_preprocessing.py --split 4 > ../data/graspXL/output4.log 2>&1 &
python grab/graspxl_preprocessing.py --split 5 > ../data/graspXL/output5.log 2>&1 &
python grab/graspxl_preprocessing.py --split 6 > ../data/graspXL/output6.log 2>&1 &
python grab/graspxl_preprocessing.py --split 7 > ../data/graspXL/output7.log 2>&1 &
python grab/graspxl_preprocessing.py --split 8 > ../data/graspXL/output8.log 2>&1 &
python grab/graspxl_preprocessing.py --split 9 > ../data/graspXL/output9.log 2>&1 &

