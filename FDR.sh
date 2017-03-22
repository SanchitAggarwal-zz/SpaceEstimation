#!/bin/bash
InputDir='../Input/Computed/RegSize60/NonClutter/Sun/'
OutputDir='../Output/RegSize60/NonClutter/Sun/'
Confidence=80
sh FDJ.sh  $InputDir $OutputDir $Confidence

InputDir='../Input/Computed/RegSize60/Clutter/Sun/'
OutputDir='../Output/RegSize60/Clutter/Sun/'
Confidence=80
sh FDJ.sh  $InputDir $OutputDir $Confidence

InputDir='../Input/Computed/RegSize60/NonClutter/CVIT/'
OutputDir='../Output/RegSize60/NonClutter/CVIT/'
Confidence=80
sh FDJ.sh  $InputDir $OutputDir $Confidence

InputDir='../Input/Computed/RegSize60/Clutter/CVIT/'
OutputDir='../Output/RegSize60/Clutter/CVIT/'
Confidence=80
sh FDJ.sh  $InputDir $OutputDir $Confidence

InputDir='../Input/Computed/RegSize60/Mixed_Sun/'
OutputDir='../Output/RegSize60/Mixed_Sun/'
Confidence=80
sh FDJ.sh  $InputDir $OutputDir $Confidence


