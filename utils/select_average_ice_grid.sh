#!/usr/bin/env bash


data_dir="/home/hpc-rosneft/nfs/110_31/NEMO-ARCT/coarse_grid"
tmpdir="/home/hpc-rosneft/drigo/surrogate/data/tmp"
outdir="/home/hpc-rosneft/nfs/0_42/share_rosneft/output_yearly/"


for year in $(seq 2010 2015); do
    echo $year
    count=1

    for i in ${data_dir}/${year}/ARCTIC_1h_ice_grid_TUV_*; do

        echo "Processing", ${year}, $count

        cdo select,levidx=1 ${i} ${tmpdir}/tmp_ice.nc
        cdo -P 16 daymean ${tmpdir}/tmp_ice.nc ${tmpdir}/${count}_ice.nc
        (( count++ ))

    done

    output=${outdir}${year}
    mkdir -p ${output}
    cdo mergetime ${tmpdir}/*_ice.nc ${output}/ice_grid_y${year}.nc

    rm ${tmpdir}/*_ice.nc
done
