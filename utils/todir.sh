#!/usr/bin/env bash
data="/Volumes/Drigo_HDD/arctic/nemo_rest/"
outdir="/Volumes/Drigo_HDD/arctic/nemo_rest_yearly/"

for i in ${data}*; do

    name=${i##*-}
    name="${name%.*}"
    #echo $name
    year=${name:0:4}
    echo $year
    #month=${name:4:2}
    #echo $month
    output=${outdir}${year}
    mkdir ${output}

    #cdo daymean ${i} ${output}/"nemo_rest_"${year}.nc
done