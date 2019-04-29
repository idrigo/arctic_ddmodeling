#!/usr/bin/env bash

tmpdir="/home/hpc-rosneft/drigo/surrogate/data/tmp"
data_dir="/home/hpc-rosneft/nfs/0_42/share_rosneft/output_yearly"


for year in 2010 2011 2012 2013 2014 2015; do
    echo ${year}
    count=1

    for i in ${data_dir}/${year}/ice_grid_y${year}.nc; do

        echo "Processing", ${year}, $count

        (( count++ ))
        cdo selyear,${year} ${i} ${i}_r
        cdo daymean ${i} ${i}_r
        mv ${i}_r ${i} -f
    done

        for i in ${data_dir}/${year}/T_grid_y${year}.nc; do

        echo "Processing", ${year}, $count

        (( count++ ))
        cdo selyear,${year} ${i} ${i}_r
        cdo daymean ${i} ${i}_r
        mv ${i}_r ${i} -f
    done

done
