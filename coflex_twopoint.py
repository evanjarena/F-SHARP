import numpy as np
import pandas as pd
from classy import Class
import pickle
import sys,os
import astropy
from astropy.cosmology import Planck15
from astropy import units as u

import matplotlib.pyplot as plt
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)

from scipy import interpolate
from scipy import integrate
from scipy import special
from scipy.signal import argrelextrema

class CoflexTwopoint:

    def __init__(self, coflex_power, survey, bin_combo):
        self.l_list = coflex_power['ell']
        self.P_F_list = coflex_power['P_F']
        self.P_kappa_F_list = coflex_power['P_kappa_F']
        self.survey = str(survey)
        self.bin_combo = str(bin_combo)

    def getTwoPoint(self):
        
        # First, interpolate all arrays so that they can be turned into callable functions
        self.interpolateArrays()
        
        # .. Get two point correlation functions
        # .. .. F-F autocorrelation
        xi_FF_plus = self.two_point_corr_flexflex(theta_flexflex_list, 'FF_plus')    
        xi_FF_minus = self.two_point_corr_flexflex(theta_flexflex_list, 'FF_minus')  
        # .. .. F-G cross-correlation. Note: xi_FG_plus = -xi_FF_minus
        xi_FG_plus = [-xi_FF_minus[i] for i in range(len(xi_FF_minus))] 
        xi_FG_minus = self.two_point_corr_flexflex(theta_flexflex_list, 'FG_minus')  
        # .. .. G-G cross correlation. Note: xi_GG_plus = xi_FF_plus
        xi_GG_plus = xi_FF_plus
        xi_GG_minus = self.two_point_corr_flexflex(theta_flexflex_list, 'GG_minus') 
        # .. Export flexion-flexion correlation functions to .pkl file
        col_list = ['theta', 'xi_FF_plus', 'xi_FF_minus', 'xi_FG_plus', 'xi_FG_minus', 'xi_GG_plus', 'xi_GG_minus']
        arrs = [theta_flexflex_list, xi_FF_plus, xi_FF_minus, xi_FG_plus, xi_FG_minus, xi_GG_plus, xi_GG_minus]
        dat = {i:arrs[j] for i,j in zip(col_list, range(len(col_list)))}
        out_frame = pd.DataFrame(data = dat, columns = col_list)
        out_frame.to_pickle(self.survey+'/'+self.survey+'_Theory/flexion-flexion_two_point_'+self.survey+'_bin_combo_'+self.bin_combo+'.pkl') 

        # Shear-flexion correlations:
        # .. Get theta_list
        theta_shearflex_list = self.theta_shearflex_list()
        # .. Get two point correlation functions
        # .. .. gam-F cross-correlation
        xi_gamF_plus = self.two_point_corr_shearflex(theta_shearflex_list, 'gamF_plus')    
        xi_gamF_minus = self.two_point_corr_shearflex(theta_shearflex_list, 'gamF_minus')  
        # .. .. G-gam cross-correlation. Note: xi_Ggam_plus = xi_gamF_minus
        xi_Ggam_plus = xi_gamF_plus
        xi_Ggam_minus = self.two_point_corr_shearflex(theta_shearflex_list, 'Ggam_minus')  
        # .. Export flexion-flexion correlation functions to .pkl file
        col_list = ['theta', 'xi_gamF_plus', 'xi_gamF_minus', 'xi_Ggam_plus', 'xi_Ggam_minus']
        arrs = [theta_shearflex_list, xi_gamF_plus, xi_gamF_minus, xi_Ggam_plus, xi_Ggam_minus]
        dat = {i:arrs[j] for i,j in zip(col_list, range(len(col_list)))}
        out_frame = pd.DataFrame(data = dat, columns = col_list)
        out_frame.to_pickle(self.survey+'/'+self.survey+'_Theory/shear-flexion_two_point_'+self.survey+'_bin_combo_'+self.bin_combo+'.pkl')

    def theta_flexflex_list(self, theta_min=1, theta_max=100, N_theta=100):
        """
        List of theta values for real-space cosmic flexion correlation functions
        Input angle values are in untis of arcseconds
        self, theta_min=1, theta_max=120, N_theta=100
        """
        # Create logspace list of angular scale, in units of arcseconds
        theta_min = np.log10(theta_min)
        theta_max = np.log10(theta_max)
        theta_list = np.logspace(theta_min,theta_max,N_theta)

        dtheta = np.log10(theta_list[1])-np.log10(theta_list[0])
        bin_low_list = 10**(np.log10(theta_list)-dtheta/2)
        bin_high_list = 10**(np.log10(theta_list)+dtheta/2)
        theta_max = np.log10(bin_high_list[-1])

        theta_list = np.logspace(theta_min,theta_max,N_theta)
        theta_list *= u.arcsec
        return theta_list

    def theta_shearflex_list(self, theta_min=1/60, theta_max=10., N_theta=100):
        """
        List of theta values for real-space cosmic shear-flexion correlation functions
        Input angle values are in untis of arcminutes
        self, theta_min=0.01, theta_max=15., N_theta=100
        theta_min=1/60, theta_max=50., N_theta=100
        """
        # Create logspace list of angular scale, in units of arcseconds
        theta_min = np.log10(theta_min)
        theta_max = np.log10(theta_max)
        theta_list = np.logspace(theta_min,theta_max,N_theta)

        dtheta = np.log10(theta_list[1])-np.log10(theta_list[0])
        bin_low_list = 10**(np.log10(theta_list)-dtheta/2)
        bin_high_list = 10**(np.log10(theta_list)+dtheta/2)
        theta_max = np.log10(bin_high_list[-1])

        theta_list = np.logspace(theta_min,theta_max,N_theta)
        theta_list *= u.arcmin
        return theta_list

    def interpolateArrays(self):
        self.P_F_interpolate = interpolate.interp1d(self.l_list, self.P_F_list)
        self.P_kappa_F_interpolate = interpolate.interp1d(self.l_list, self.P_kappa_F_list)

    def P_F(self, ell):
        return self.P_F_interpolate(ell)

    def P_kappa_F(self, ell):
        return self.P_kappa_F_interpolate(ell)

    def two_point_corr_flexflex(self, theta_list, fields):

        # First, convert the list of angles to radians
        theta_list_rad = theta_list.to(u.rad).value

        # Get parameters specific to the particular two-point correlation function. 
        #  These include the order of the Bessel function for the Hankel transform,
        #  as well as the algebraic sign of the two-point correlation function.
        if fields == 'FF_plus':
            order = 0
            sign = (+1)
        elif fields == 'FF_minus':
            order = 2
            sign = (-1)
        elif fields == 'FG_plus':
            order = 2
            sign = (+1)
        elif fields == 'FG_minus':
            order = 4
            sign = (-1)
        elif fields == 'GG_plus':
            order = 0
            sign = (+1)
        elif fields == 'GG_minus':
            order = 6
            sign = (-1)

        # Get two-point correlation function for each angular separation
        xi_list_norm = []
        for theta in theta_list_rad:
            # Get down-sampled ell list
            l_list = np.logspace(np.log10(np.min(self.l_list)), np.log10(np.max(self.l_list)), int(1e7))
            
            # Get integrand of two-point correlation function
            xi_integrand_unnorm = l_list * special.jv(order, l_list*theta)*self.P_F(l_list)

            # Perform integrand renormalization.
            ell_min_index = argrelextrema(xi_integrand_unnorm, np.less)[0]
            ell_min = l_list[ell_min_index]
            xi_integrand_min = xi_integrand_unnorm[ell_min_index]
            id_min = np.where(xi_integrand_min < 0)
            ell_min_1 = ell_min[id_min][0]
            ell_max_index = argrelextrema(xi_integrand_unnorm, np.greater)[0]
            ell_max = l_list[ell_max_index][0]
            ell_max_1 = l_list[ell_max_index][1]
            ell_max_2 = l_list[ell_max_index][2]
            xi_integrand_norm = xi_integrand_unnorm*np.e**(-((l_list*(l_list+1))/ell_max_2**2.))
            
            # Now we can integrate.  We use Simpson's rule for fast integration 
            xi_integral_norm = integrate.simps(xi_integrand_norm, l_list, axis=-1)
            # xi = 1/2pi times the integral, with the appropriate algebraic sign
            xi_norm = sign*(1/(2*np.pi))*xi_integral_norm
            xi_list_norm.append(xi_norm)
            
        return xi_list_norm

    def two_point_corr_shearflex(self, theta_list, fields):

        # First, convert the list of angles to radians
        theta_list_rad = theta_list.to(u.rad).value

        # Get parameters specific to the particular two-point correlation function. 
        #  These include the order of the Bessel function for the Hankel transform,
        #  as well as the algebraic sign of the two-point correlation function.
        if fields == 'gamF_plus':
            order = 1
            sign = (-1)
        elif fields == 'gamF_minus':
            order = 3
            sign = (+1)
        elif fields == 'Ggam_plus':
            order = 1
            sign = (-1)
        elif fields == 'Ggam_minus':
            order = 5
            sign = (-1)

        # Get two-point correlation function for each angular separation
        xi_list_norm = []
        for theta in theta_list_rad:
            # Get down-sampled ell list
            l_list = np.logspace(np.log10(np.min(self.l_list)), np.log10(np.max(self.l_list)), int(1e7))

            # Get integrand of two-point correlation function
            xi_integrand_unnorm = l_list * special.jv(order, l_list*theta)*self.P_kappa_F(l_list)

            # Perform integrand renormalization.
            ell_min_index = argrelextrema(xi_integrand_unnorm, np.less)[0]
            ell_min = l_list[ell_min_index]
            xi_integrand_min = xi_integrand_unnorm[ell_min_index]
            id_min = np.where(xi_integrand_min < 0)
            ell_min_1 = ell_min[id_min][0]
            ell_max_index = argrelextrema(xi_integrand_unnorm, np.greater)[0]
            ell_max = l_list[ell_max_index][0]
            ell_max_1 = l_list[ell_max_index][1]
            ell_max_2 = l_list[ell_max_index][2]
            xi_integrand_norm = xi_integrand_unnorm*np.e**(-((l_list*(l_list+1))/ell_max_2**2.))
            
            # Now we can integrate.  We use Simpson's rule for fast integration 
            xi_integral_norm = integrate.simps(xi_integrand_norm, l_list, axis=-1)
            # xi = 1/2pi times the integral, with the appropriate algebraic sign
            xi_norm = sign*(1/(2*np.pi))*xi_integral_norm
            xi_list_norm.append(xi_norm)
            
        return xi_list_norm


























        
