#!/bin/bash
matlab -nodesktop -nosplash -singleCompThread -r "ComputeImageCues('$1','$2',$3), exit"

