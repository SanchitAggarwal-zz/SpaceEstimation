#!/bin/bash
matlab -nodesktop -nosplash -singleCompThread -r "FloorDetection('$1','$2',$3), exit"

