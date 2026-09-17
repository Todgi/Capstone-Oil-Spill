#!/usr/bin/env python
"""
Openoil - Laut Jawa
==================================
"""

from datetime import datetime, timedelta
from opendrift.readers import reader_netCDF_CF_generic
from opendrift.models.openoil import OpenOil

o = OpenOil(loglevel=20, location='Indonesia')

print(o.oiltypes)  # Print available oil types

#%% Add forcing data
# Using global data sources that cover Indonesian waters
# Global weather forecast from NCEP
reader_ncep = reader_netCDF_CF_generic.Reader('https://pae-paha.pacioos.hawaii.edu/thredds/dodsC/ncep_global/NCEP_Global_Atmospheric_Model_best.ncd')
# Global ocean model from HYCOM
reader_hycom = reader_netCDF_CF_generic.Reader('https://tds.hycom.org/thredds/dodsC/FMRC_ESPC-D-V02_uv3z/FMRC_ESPC-D-V02_uv3z_best.ncd')
o.add_reader([reader_hycom, reader_ncep])

#%%
# Adjusting some configuration
o.set_config('processes:evaporation',  True)
o.set_config('processes:emulsification',  True)
o.set_config('drift:vertical_mixing',  True)
o.set_config('vertical_mixing:timestep',  5)

#%%
# Seeding some particles
time = datetime.now()  # Use current time instead of reader start time
oil_type = 'SUMATRAN LIGHT'  # Menggunakan jenis minyak yang tersedia dari Indonesia
o.seed_elements(lon=106.192765, lat=-5.082163, radius=3000, number=2000,
                time=time, z=0, oil_type=oil_type)

#%%
# Running model
o.run(steps=4*40, time_step=900, time_step_output=3600)

#%%
# Print and plot results
print(o)
o.plot(fast=True)
o.plot_oil_budget()
#o.plot(filename='openoil_drift_laut_jawa')
o.plot_vertical_distribution(maxnum=100,bins=50)
o.plot_property('water_fraction', mean=True)
o.plot_property('z')
#o.plot_property('mass_evaporated')
#o.plot_property('water_fraction')
#o.plot_property('interfacial_area')
o.animation(fast=True)

#%%
# .. image:: /gallery/animations/example_openoil_0.gif
