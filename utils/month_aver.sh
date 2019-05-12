#!/usr/bin/env bash

tmpdir="/home/hpc-rosneft/drigo/surrogate/data/tmp"
data_dir="/home/hpc-rosneft/nfs/0_42/share_rosneft/output_yearly"


for year in 2010 2011 2012 2013 2014 2015 2016; do
    echo ${year}
    count=1
    cp ${data_dir}/${year}/ice-thick-cr2smos-${year}.nc ${tmpdir}/ice-thick-cr2smos-${year}.nc
done