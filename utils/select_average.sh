#!/usr/bin/env bash


data_dir="/home/hpc-rosneft/nfs/110_31/NEMO-ARCT/coarse_grid"
tmpdir="/home/hpc-rosneft/drigo/surrogate/data/tmp"
outdir="/home/hpc-rosneft/nfs/0_42/share_rosneft/output_yearly/"


for year in $(seq 2010 2015); do
    echo $year
    count=1

    for i in ${data_dir}/${year}/ARCTIC_1h_T_grid_T_*; do

        #year="$(cut -d'/' -f8 <<<"$i")"
        echo "Processing", ${year}, $count
        output=${outdir}${year}
        mkdir -p ${output}

        #cdo select,name=sossheig,vosaline,votemper,level=0 ${i} ${tmpdir}/tmp.nc
        #cdo daymean ${tmpdir}/tmp.nc ${output}/T_grid_y${year}.nc

        #cdo select,name=iceconc,icethic_cea,snowthic_cea,uice_ipa,vice_ipa,level=0 ${i}/ARCTIC_1h_ice_grid_TUV_*.nc ${tmpdir}/tmp.nc
        #cdo daymean ${tmpdir}/tmp.nc ${output}/ice_grid_TUV_y${year}.nc


        cdo -P 16 -sellevidx,1 -daymean ${i} ${tmpdir}/${count}.nc
        (( count++ ))

    done

    cdo mergetime ${tmpdir}/*.nc ${output}/T_grid_y${year}.nc

    rm ${tmpdir}/*
done
