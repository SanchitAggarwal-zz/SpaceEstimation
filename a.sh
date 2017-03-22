#!/bin/sh
#$ -cwd
callmatlab(){
	matlab -nojvm -nosplash -nodisplay -r getClassificationModel($1,$2,$3,$4,$5,$6) 
}

datafile='/Pulsar3/sanchit.aggarweal/Code/data.mat'
split=1
nfold=5
for svm_type in 0 3 4
do
	for kernel_type in 1 2 3
   	do
		cost=-20
		while [ $cost -le 20 ]
		do
	  		gamma=-20
			while [ $gamma -le 20 ]
			do
				qsub callmatlab $datafile $split $nfold $svm_type $kernel_type $cost $gamma
				gamma=$((gamma + 1))
			done
			cost=$((cost + 1))
		done
	done
done

			 
