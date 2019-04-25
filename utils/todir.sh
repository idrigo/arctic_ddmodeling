#!/usr/bin/env bash
data="/home/hpc-rosneft/nfs/0_42/share_rosneft/rean_int"
outdir="/home/hpc-rosneft/nfs/0_42/share_rosneft/output_yearly/"

for i in ${data}/*; do

    name="$(cut -d'/' -f8 <<<"$i")"
    year=${name:(-7):(-3)}
    echo $year
    output=${outdir}${year}
    mkdir -p ${output}


    cdo daymean ${i} ${output}/${name}
done