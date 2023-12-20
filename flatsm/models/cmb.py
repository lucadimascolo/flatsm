from .utils import *

# ======================================================================
# Random CMB realization
# ----------------------------------------------------------------------
class getcmb:
  def __init__(self,shape,wcs,cosmo):  
    pars = camb.CAMBparams()
    pars.set_cosmology(H0    =  cosmo.H0.to(u.km/u.s/u.Mpc).value,
                       tau   =  0.0540,
                       omch2 = (cosmo.Om0-cosmo.Ob0)*cosmo.h**2,
                       ombh2 =  cosmo.Ob0*cosmo.h**2,
                       TCMB  =  cosmo.Tcmb0.to(u.K).value,
                       standard_neutrino_neff = cosmo.Neff,
                       mnu   = cosmo.m_nu.to(u.eV).value[0]*np.floor(cosmo.Neff))

    pars.set_for_lmax(20000,lens_potential_accuracy=1)

    res = camb.get_results(pars)
    pow = res.get_cmb_power_spectra(pars,CMB_unit='K',raw_cl=True)

    TT, EE, BB, TE = pow['unlensed_scalar'].T

    powcmb = np.array([TT,TE,np.zeros(TT.shape),EE,np.zeros(TT.shape),BB])
    
    self.model = self.smallscales(shape,wcs,powcmb)
    self.sky   = None

# Generate CMB map
# ------------------------------
  def smallscales(self,shape,wcs,cls):
    chead = wcs.to_header()
    csize = (shape[0]*np.abs(chead['CDELT2'])*u.deg,
             shape[1]*np.abs(chead['CDELT1'])*u.deg)
    del chead

    imgcmb = pymaster.synfast_flat(nx = shape[1], lx = np.deg2rad(csize[1].to(u.deg).value),
                                   ny = shape[0], ly = np.deg2rad(csize[0].to(u.deg).value), cls = cls, spin_arr = [0,2])
    return imgcmb*u.K_CMB

# Provide frequency map
# ------------------------------
  def get_emission(self,freqs,weights=None):
    return self.model