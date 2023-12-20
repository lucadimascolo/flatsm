from .utils import *

# ======================================================================
# Hybrid Galactic synchrotron realization
# ----------------------------------------------------------------------
class getsynch:
  def __init__(self,sky,shape,wcs,rot,seeds,**kwargs):
    self.sky = sky
    self.rot = rot

    self.model = self.smallscales(shape,wcs,seeds,**kwargs)

# Coordinate rotation
# ------------------------------
  def alm2map(self,alm,nside):
    return healpy.alm2map(self.rot.rotate_alm(alm),nside)
  
# Power law spectral scaling
# ------------------------------
  def powlaw(self,freqs,weights,component='I'):
    freqs   = pysm3.utils.check_freq_input(freqs)
    weights = pysm3.utils.normalize_weights(freqs,weights)

    reffreq = getattr(self.sky,f'freq_ref_{component}').value

    delfreq = 0.50*np.concatenate([np.diff(freqs),np.diff(freqs)[-1:]]) if freqs.shape[0]>1 else np.array([1.00])

    outspec = []
    for fi, f in enumerate(freqs):
      outspec.append((f/reffreq)**self.model['spix'].value)
      outspec[fi] *= delfreq[fi]*weights[fi]
    return np.sum(outspec,axis=0) if len(outspec)>1 else outspec[0]

# Generate frequency maps
# ------------------------------
  def get_emission(self,freqs,weights=None):
    freq_scaling_I = self.powlaw(freqs,weights,'I')
    
    if self.model['Q'] is not None:
      freq_scaling_P = self.powlaw(freqs,weights,'P')
      output = np.array([self.model['I'].value*freq_scaling_I,
                         self.model['Q'].value*freq_scaling_P,
                         self.model['U'].value*freq_scaling_P])
    else:
      output = np.array([self.model['I'].value*freq_scaling_I])

    return output*u.uK_RJ

# re-calculating small scales
# ------------------------------
  def smallscales(self,shape,wcs,seeds,**kwargs):
    galplane_fix_map = kwargs.get('galplane_fix',False)

    chead = wcs.to_header()
    csize = (shape[0]*np.abs(chead['CDELT2'])*u.deg,
             shape[1]*np.abs(chead['CDELT1'])*u.deg)
    del chead

  # modulation maps
    map_modulation = []
    for mi in range(len(self.sky.modulate_alm)):
      tmpmap = self.alm2map(self.sky.modulate_alm[mi].value,self.sky.nside)
    
      tmpmod = pixell.enmap.zeros((1,*shape),wcs=wcs)
      tmpmod = pixell.reproject.healpix2map(tmpmap,out=tmpmod,method='spline')[0]
      map_modulation.append(np.asarray(tmpmod)); del tmpmap, tmpmod

    pli_modulation = map_modulation[2 if len(self.sky.modulate_alm)==3 else 0]

  # large-scale T,Q,U template
    map_template = []
    raw_template = self.alm2map(self.sky.template_largescale_alm.value,self.sky.nside)

    map_template = pixell.enmap.zeros((3,*shape),wcs=wcs)
    map_template = pixell.reproject.healpix2map(raw_template,out=map_template,method='spline')
    map_template = np.asarray(map_template)

  # small-scale I,Q,U maps
    TT, EE, BB, TE = self.sky.small_scale_cl.real.value
    cls_small_scale = np.array([TT,TE,np.zeros(TT.shape),EE,np.zeros(TT.shape),BB])

    map_small_scale = pymaster.synfast_flat(nx = shape[1], lx = np.deg2rad(csize[1].to(u.deg).value),
                                            ny = shape[0], ly = np.deg2rad(csize[0].to(u.deg).value), 
                                           cls = cls_small_scale, spin_arr = [0,2], seed = seeds[0])
    del cls_small_scale

    for mi in range(map_small_scale.shape[0]):
      map_small_scale[mi] *= map_modulation[int(mi!=0)]
      map_small_scale[mi] += map_template[mi]

    map_small_scale = pysm3.utils.log_pol_tens_to_map(map_small_scale)*self.sky.template_largescale_alm.unit

    if galplane_fix_map and (self.sky.galplane_fix_map is not None):
      gal_small_scale = pixell.enmap.zeros((4,*shape),wcs=wcs)
      gal_small_scale = pixell.reproject.healpix2map(self.sky.galplane_fix_map[:4].value,out=gal_small_scale,method='spline')
    
      map_small_scale *= gal_small_scale[3]
      map_small_scale += gal_small_scale[:3]*(1-gal_small_scale[3])*self.sky.galplane_fix_map.unit

    pli_small_scale = {}
      
    alm_large_scale = getattr(self.sky,f'largescale_alm_pl_index')
    sky_large_scale = self.alm2map(alm_large_scale.value,self.sky.nside)
    
    mbb_large_scale = pixell.enmap.zeros((1,*shape),wcs=wcs)
    mbb_large_scale = pixell.reproject.healpix2map(sky_large_scale,out=mbb_large_scale,method='spline')[0]

    cls_small_scale = getattr(self.sky,f'small_scale_cl_pl_index')

    pli_small_scale['spix'] = pymaster.synfast_flat(nx = shape[1], lx = np.deg2rad(csize[1].to(u.deg).value),
                                                    ny = shape[0], ly = np.deg2rad(csize[0].to(u.deg).value), 
                                                   cls = [cls_small_scale.real.value], 
                                              spin_arr = [0], seed = seeds[1])[0]

    pli_small_scale['spix']  = pli_small_scale['spix']*pli_modulation+np.asarray(mbb_large_scale)
    pli_small_scale['spix'] *= np.sqrt(1*cls_small_scale.unit).unit
    pli_small_scale['spix'] -= 3.10*u.dimensionless_unscaled
    del mbb_large_scale, sky_large_scale, cls_small_scale
   
    pli_small_scale['I'] = map_small_scale[0].copy()
    pli_small_scale['Q'] = map_small_scale[1].copy()
    pli_small_scale['U'] = map_small_scale[2].copy()

    return pli_small_scale