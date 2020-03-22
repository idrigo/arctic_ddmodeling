local = True
if local:
    var_dict = [{'variable': 'Thickness',  # sattelite
                 'path': '/Volumes/Drigo_HDD/arctic/ice-restoring/',
                 'file_mask': 'ice_thick_y{}.nc'},

                {'variable': 'thick_cr2smos',  # sattelite
                 'path': '/Volumes/Drigo_HDD/arctic/output_yearly/',
                 'file_mask': 'ice-thick-cr2smos-{}.nc'},

                {'variable': 'ice_conc',  # sattelite ice concecteation
                 'path': '/Volumes/Drigo_HDD/arctic/ice-restoring/',
                 'file_mask': 'conc_y{}.nc'},

                {'variable': 'radlw',  # reanalysis LW radiation
                 'path': '/Volumes/Drigo_HDD/arctic/forcing_auto/',
                 'file_mask': 'dfs_radlwf_y{}.nc'},

                {'variable': 'radsw',  # reanalysis SW radiation
                 'path': '/Volumes/Drigo_HDD/arctic/forcing_auto/',
                 'file_mask': 'dfs_radswf_y{}.nc'},

                {'variable': 'tair',  # reanalysis air temperature
                 'path': '/Volumes/Drigo_HDD/arctic/output_yearly/',
                 'file_mask': 'arctic_tair_y{}.nc'},

                {'variable': 'iceconc',  # model ice concentration
                 'path': '/Volumes/Drigo_HDD/arctic/output_yearly/',
                 'file_mask': 'ice_grid_y{}.nc'},

                {'variable': 'icethic_cea',  # model ice thickness
                 'path': '/Volumes/Drigo_HDD/arctic/output_yearly/',
                 'file_mask': 'ice_grid_y{}.nc'},

                {'variable': 'icethic_cea_LASSO',  # model ice thickness with LASSO assimilation
                 'path': '/Volumes/Drigo_HDD/arctic/output_yearly/',
                 'file_mask': 'ice_grid_y{}_LASSO.nc'},

                {'variable': 'snowthic_cea',  # model snow thickness
                 'path': '/Volumes/Drigo_HDD/arctic/output_yearly/',
                 'file_mask': 'ice_grid_y{}.nc'},

                {'variable': 'vosaline',  # model salinity
                 'path': '/Volumes/Drigo_HDD/arctic/output_yearly/',
                 'file_mask': 'T_grid_y{}.nc'},

                {'variable': 'votemper',  # model pot temperature
                 'path': '/Volumes/Drigo_HDD/arctic/output_yearly/',
                 'file_mask': 'T_grid_y{}.nc'},

                {'variable': 'sossheig',  # model SSH
                 'path': '/Volumes/Drigo_HDD/arctic/output_yearly/',
                 'file_mask': 'T_grid_y{}.nc'},

                {'variable': 'hice',  # TOPAZ reanalysis ice height
                 'path': '/Volumes/Drigo_HDD/arctic/output_yearly/',
                 'file_mask': 'TOPAZ_y{}.nc'}
                ]
    mask_path = '/Volumes/Drigo_HDD/arctic/mask.npy'
    processed_data_path = '/Volumes/Drigo_HDD/arctic/data_numpy_dump/'
else:
    var_dict = [{'variable': 'Thickness',  # sattelite
                 'path': '/home/hpc-rosneft/nfs/0_42/share_rosneft/input_data/NEMO/14km/ice-restoring/',
                 'file_mask': 'ice_thick_y{}.nc'},

                {'variable': 'thick_cr2smos',  # sattelite
                 'path': '/home/hpc-rosneft/nfs/0_42/share_rosneft/output_yearly/',
                 'file_mask': 'ice-thick-cr2smos-{}.nc'},

                {'variable': 'ice_conc',  # sattelite ice concecteation
                 'path': '/home/hpc-rosneft/nfs/0_42/share_rosneft/input_data/NEMO/14km/ice-restoring/',
                 'file_mask': 'conc_y{}.nc'},

                {'variable': 'radlw',  # reanalysis LW radiation
                 'path': '/home/hpc-rosneft/nfs/0_42/share_rosneft/input_data/NEMO/14km/forcing_auto/',
                 'file_mask': 'dfs_radlwf_y{}.nc'},

                {'variable': 'radsw',  # reanalysis SW radiation
                 'path': '/home/hpc-rosneft/nfs/0_42/share_rosneft/input_data/NEMO/14km/forcing_auto/',
                 'file_mask': 'dfs_radswf_y{}.nc'},

                {'variable': 'tair',  # reanalysis air temperature
                 'path': '/home/hpc-rosneft/nfs/0_42/share_rosneft/output_yearly/',
                 'file_mask': 'arctic_tair_y{}.nc'},

                {'variable': 'iceconc',  # model ice concentration
                 'path': '/home/hpc-rosneft/nfs/0_42/share_rosneft/output_yearly/',
                 'file_mask': 'ice_grid_y{}.nc'},

                {'variable': 'icethic_cea',  # model ice thickness
                 'path': '/home/hpc-rosneft/nfs/0_42/share_rosneft/output_yearly/',
                 'file_mask': 'ice_grid_y{}.nc'},

                {'variable': 'snowthic_cea',  # model snow thickness
                 'path': '/home/hpc-rosneft/nfs/0_42/share_rosneft/output_yearly/',
                 'file_mask': 'ice_grid_y{}.nc'},

                {'variable': 'vosaline',  # model salinity
                 'path': '/home/hpc-rosneft/nfs/0_42/share_rosneft/output_yearly/',
                 'file_mask': 'T_grid_y{}.nc'},

                {'variable': 'votemper',  # model pot temperature
                 'path': '/home/hpc-rosneft/nfs/0_42/share_rosneft/output_yearly/',
                 'file_mask': 'T_grid_y{}.nc'},

                {'variable': 'sossheig',  # model SSH
                 'path': '/home/hpc-rosneft/nfs/0_42/share_rosneft/output_yearly/',
                 'file_mask': 'T_grid_y{}.nc'}
                ]
    mask_path = '/home/hpc-rosneft/drigo/surrogate/data/mask.npy'
    processed_data_path = '/home/hpc-rosneft/nfs/0_42/share_rosneft/data_numpy_dump/'
