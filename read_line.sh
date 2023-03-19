#!/bin/bash
input="/scratch/cg3306/climate/code/jobs/filter_weights.txt"
i=0
j=2
ARGS=0
while IFS= read -r line
do
    if [[ "$i" == "$j" ]]
    then 
        ARGS="$line"
        break
    fi
    ((i++))
done < "$input"
echo $ARGS