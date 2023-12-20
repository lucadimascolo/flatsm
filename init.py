from astropy import units as u
from astropy.io import fits

import flatsm
import numpy as np
import dill

bfile = '../support/bands.pickle'
with open(bfile,'rb') as bload:
  bands = dill.load(bload)

crval1 = 0.00
crval2 = 0.00

cdelt = 1.00*u.arcsec
csize = 1.00*u.deg

flist = [np.mean(bands[band]['freq'])*u.GHz for band in bands.keys()]
flist = [flist[0]]

sky = flatsm.build(csize,cdelt,pos=(crval1,crval2))
map_output = np.array([sky.get_emission(f).value for f in flist])

fits.writeto('test_I.fits',map_output[:,0,:,:],overwrite=True)