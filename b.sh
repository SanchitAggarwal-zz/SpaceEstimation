#!/bin/bash
power(){
	local n="$1"
	local k="$2"
	local res=$n
	while [ $k -le 0 ]
	do
		res=$($bc -l <<< "$res / $n")
		k=$((k + 1)) 
	done

	echo $res
}

x=-20
while [ $x -le 20 ]
do
	  if [ $x -gt -1 ]
		then
  		$powq=$((2 ** $x))
	else
  		 power 2 $x
	fi
echo $powq
done


