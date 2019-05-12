#!/usr/bin/env bash


data_dir="/home/hpc-rosneft/nfs/110_31/NEMO-ARCT/coarse_grid"
tmpdir="/home/hpc-rosneft/drigo/surrogate/data/tmp"
outdir="/home/hpc-rosneft/nfs/0_42/share_rosneft/output_yearly/"


for year in 2011 2012; do
    echo ${year}
    count=1
    rm ${tmpdir}/*_T.nc
    for i in ${data_dir}/${year}/ARCTIC_1h_T_grid_T_*; do

        echo "Processing", ${year}, $count

        cdo -P 16 select,name=votemper,vosaline,levidx=1 ${i} ${tmpdir}/tmp_T.nc
        cdo -P 16 daymean ${tmpdir}/tmp_T.nc ${tmpdir}/${count}_T.nc
        (( count++ ))

    done
    output=${outdir}${year}
    mkdir -p ${output}
    cdo mergetime ${tmpdir}/*_T.nc ${output}/T_grid_y${year}.nc

    printf 'Press Enter to continue'
    read REPLY
done
