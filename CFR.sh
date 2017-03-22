#!/bin/bash
InputDir='../ICPR_DataSet/Sun/NonClutter/'
OutputDir='../Input/Computed/RegSize60/NonClutter/Sun/'
RegSize=60
sh CFJ.sh  $InputDir $OutputDir $RegSize

InputDir='../ICPR_DataSet/CVIT/NonClutter/'
OutputDir='../Input/Computed/RegSize60/NonClutter/CVIT/'
RegSize=60
sh CFJ.sh  $InputDir $OutputDir $RegSize

InputDir='../ICPR_DataSet/Sun/Clutter/'
OutputDir='../Input/Computed/RegSize60/Clutter/Sun/'
RegSize=60
sh CFJ.sh  $InputDir $OutputDir $RegSize

InputDir='../ICPR_DataSet/CVIT/Clutter/'
OutputDir='../Input/Computed/RegSize60/Clutter/CVIT/'
RegSize=60
sh CFJ.sh  $InputDir $OutputDir $RegSize


InputDir='../ICPR_DataSet/Mixed_Sun/'
OutputDir='../Input/Computed/RegSize60/Mixed_Sun/'
RegSize=60
sh CFJ.sh  $InputDir $OutputDir $RegSize

