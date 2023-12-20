import pysm3
import pysm3.units as u

import numpy as np

import healpy

import pixell.enmap
import pixell.reproject

import pymaster
import camb

rotate = healpy.rotator.Rotator(rot=(0.00,9.00,0.00),coord=['G','E'])

def alm2map(alm,nside):
  return healpy.alm2map(rotate.rotate_alm(alm),nside)