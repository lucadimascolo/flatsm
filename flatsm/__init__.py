from astropy.cosmology import Planck18

import healpy
import pixell.enmap
import pixell.reproject

import pysm3
import pysm3.units as u

import numpy as np
import os

from . import models

os.environ['PYSM_LOCAL_DATA'] = '../support/pysm'

# Flat-sky geometry helper
# ------------------------------
def geometry(csize,cdelt,pos):
  caxis = csize/cdelt
  caxis = caxis.to(u.dimensionless_unscaled).value
  caxis = int(caxis)

  return pixell.enmap.geometry(pos = pos, 
                               res = np.deg2rad(cdelt.to(u.deg).value),
                              proj = 'car',
                             shape = (caxis,caxis))

# ======================================================================
# Generate simulated foreground maps
# ----------------------------------------------------------------------
_preset = {'dust': {'preset': 'd11', 'seeds': [8192,777,888], 'foo': models.getdust},
          'radio': {'preset':  's6', 'seeds': [698,14708],    'foo': models.getsynch}}

class build:
  def __init__(self,csize,cdelt,pos=(0,0),nside=512,preset=_preset,cosmo=Planck18,**kwargs):
    self.shape, self.wcs = geometry(csize,cdelt,(0.00,0.00))
    self.nside = nside

    self.rot = healpy.rotator.Rotator(rot=(pos[0],pos[1],0.00),coord=['G','E'])

    self.models = {}
    self.preset = []
    if len(preset):
      for ki, key in enumerate(['dust','radio']):
        if key in preset.keys(): 
          self.preset.append(preset[key]['preset'])
          self.models[key] = {'id': ki}
    
      if len(self.preset): 
        self.template = pysm3.Sky(nside=nside,preset_strings=self.preset,output_unit='K_CMB')

        for ki, key in enumerate(self.models.keys()):
          print(f'Generating {key} model')
          self.models[key]['model'] = preset[key]['foo'](self.template.components[ki],self.shape,self.wcs,self.rot,preset[key]['seeds'],**kwargs)

      if 'cmb' in preset.keys():
        self.models['cmb'] = {'model': models.getcmb(self.shape,self.wcs,cosmo)}
      
    print(f'Generating free-free and AME models')
    self.extras = pysm3.Sky(nside=nside,preset_strings=['f1','a1'],output_unit='K_CMB')
    
# Coordinate rotation
# ------------------------------
  def alm2map(self,alm,nside):
    return healpy.alm2map(self.rotate.rotate_alm(alm),nside)

# Generate total frequency maps    
# ------------------------------
  def get_emission(self,freqs,weights=None):
  # sky_output = self.extras.get_emission(freqs,weights).value
  # sky_output = healpy.map2alm(sky_output,use_pixel_weights=True)
  # sky_output = self.alm2map(sky_output,self.nside)

    map_output = pixell.enmap.zeros((3,*self.shape),wcs=self.wcs)
  # map_output = pixell.reproject.healpix2map(sky_output,out=map_output,method='spline')
    map_output *= u.K_CMB

    for ki, key in enumerate(self.models.keys()):
      factor  = self.models[key]['model'].get_emission(freqs,weights)
      factor *= pysm3.utils.bandpass_unit_conversion(freqs,weights,self.template.output_unit) if key!='cmb' else 1.00
      map_output += factor; del factor

    return map_output