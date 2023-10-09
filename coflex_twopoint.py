import numpy as np
import pandas as pd
#from classy import Class
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

##TO DO: Try cutting off integrand after SECOND (or third) time it crosses zero.  We can compare to full-blown integration and see which one is correct.

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
        #"""
        # Flexion-flexion correlations:
        # .. Get theta_list
        theta_flexflex_list = self.theta_flexflex_list(theta_min=1., theta_max=1000., N_theta=10)
        xi_FF_plus = self.two_point_corr_flexflex(theta_flexflex_list, 'FF_plus')
        print(theta_flexflex_list)
        print((xi_FF_plus))
        plt.plot(theta_flexflex_list.value, xi_FF_plus)
        plt.show()

        print('should not see this')

        # Inspect xi_FF_plus over long range
        theta_flexflex_list_test = self.theta_flexflex_list()
        
        # .. Run nonconvergence demo
        self.two_point_corr_flexflex_nonconvergence_demo(theta_flexflex_list, 'FF_minus')
        
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
        #"""
        # Shear-flexion correlations:
        # .. Get theta_list
        theta_shearflex_list = self.theta_shearflex_list()
        # .. Get two point correlation functions
        # .. .. gam-F cross-correlation
        xi_gamF_plus = self.two_point_corr_shearflex(theta_shearflex_list, 'gamF_plus')    
        xi_gamF_minus = self.two_point_corr_shearflex(theta_shearflex_list, 'gamF_minus')  
        # .. .. gam-G cross-correlation. Note: xi_gamG_plus = xi_gamF_minus
        xi_gamG_plus = xi_gamF_plus
        xi_gamG_minus = self.two_point_corr_shearflex(theta_shearflex_list, 'gamG_minus')  
        # .. Export flexion-flexion correlation functions to .pkl file
        col_list = ['theta', 'xi_gamF_plus', 'xi_gamF_minus', 'xi_gamG_plus', 'xi_gamG_minus']
        arrs = [theta_shearflex_list, xi_gamF_plus, xi_gamF_minus, xi_gamG_plus, xi_gamG_minus]
        dat = {i:arrs[j] for i,j in zip(col_list, range(len(col_list)))}
        out_frame = pd.DataFrame(data = dat, columns = col_list)
        out_frame.to_pickle(self.survey+'/'+self.survey+'_Theory/shear-flexion_two_point_'+self.survey+'_bin_combo_'+self.bin_combo+'.pkl')

        #self.xi_F_plus_integrand_nonconvergence_demo(theta_list)
        #self.xi_F_minus_integrand_nonconvergence_demo(theta_list)
        #"""
        #self.xi_F_plus_conv_demo(theta_lit)

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
        theta_list_rad = theta_list.to(u.rad).value#[90:99]

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
        xi_list_unnorm = []
        xi_list_norm = []
        for theta in theta_list_rad:
            # Get down-sampled ell list
            l_list = np.logspace(np.log10(np.min(self.l_list)), np.log10(np.max(self.l_list)), int(1e7)) #1e6, 1e8 was the last run

            # Get integrand of two-point correlation function
            xi_integrand_unnorm = l_list * special.jv(order, l_list*theta)*self.P_F(l_list)
            # Get integral
            xi_integral_unnorm = integrate.simps(xi_integrand_unnorm, l_list, axis=-1)
            #xi_unnorm = sign*(1/(2*np.pi))*xi_integral_unnorm

            #xi_list_unnorm.append(xi_unnorm)

            # Perform integrand renormalization.
            # .. The most simple and robust renormalization is to locate the first integrand peak,
            #    find the next ell value for which the interand = 0, and make this the upper bound 
            #    of integration.  Analytically, the rest of the integral should evaluate to zero
            ell_min_index = argrelextrema(xi_integrand_unnorm, np.less)[0]
            ell_min = l_list[ell_min_index]
            xi_integrand_min = xi_integrand_unnorm[ell_min_index]
            id_min = np.where(xi_integrand_min < 0)
            ell_min_1 = ell_min[id_min][0]
            ell_max_index = argrelextrema(xi_integrand_unnorm, np.greater)[0]
            ell_max = l_list[ell_max_index][0]
            ell_max_1 = l_list[ell_max_index][1]
            ell_max_2 = l_list[ell_max_index][2]
            if ell_max_1 > ell_min_1:
                ell_max_2 = ell_max_1
            l_list_int_index = np.where((l_list > ell_min_1) & (l_list < ell_max_2))
            l_list_int = l_list[l_list_int_index]
            l_cutoff_index = np.argmin(abs(xi_integrand_unnorm[l_list_int_index]))
            ell_cutoff = l_list_int[l_cutoff_index]
            id = np.where(l_list > ell_cutoff)
            xi_integrand_norm = xi_integrand_unnorm*np.e**(-((l_list*(l_list+1))/ell_max_2**2.))
            # Now we can integrate.  We use Simpson's rule for fast integration 
            xi_integral_norm = integrate.simps(xi_integrand_norm, l_list, axis=-1)
            # xi = 1/2pi times the integral, with the appropriate algebraic sign
            xi_norm = sign*(1/(2*np.pi))*xi_integral_norm
            xi_list_norm.append(xi_norm)
        return xi_list_norm































    def xi_F_plus_conv_demo(self, theta_list):

        #del_l = 1e9

        # First, get a sample integrand for plotting purposes:

        # .. First integrad: no damping
        # .. .. Choose an arbitrary angle, say -- 10 arcseconds.
        theta_arcsec = 10*u.arcsec
        # .. .. Convert to radians
        theta = theta_arcsec.to(u.rad).value
        # .. .. Temporary l_list
        l_list = np.logspace(np.log10(np.min(self.l_list)), np.log10(np.max(self.l_list)), 1e6) 
        # .. .. Get temporary integrand of xi_F_plus
        integrand = l_list * self.J0(l_list*theta)*self.P_F(l_list)
        # .. .. Get the extrema of the integrand
        l_max_indices = argrelextrema(integrand, np.greater)[0]
        l_max = np.array(([l_list[i] for i in l_max_indices]))
        print(np.shape(l_max), np.shape(l_list))
        l_min_indices = argrelextrema(integrand, np.less)[0]
        l_min = np.array(([l_list[i] for i in l_min_indices]))
        # .. .. Decrease the resolution of the l_list
        l_list = np.logspace(np.log10(np.min(self.l_list)), np.log10(np.max(self.l_list)), 1e5) 
        # .. .. Get combined l_list
        l_list = np.sort(np.hstack((l_list,l_max,l_min)))#np.sort(np.array(list(set(np.stack(l_list, l_max, l_min).flatten()))))
        # .. .. Get new interand
        integrand = l_list * self.J0(l_list*theta)*self.P_F(l_list)

        # .. Second integrand: damped after first peak
        l_peak = l_max[0]
        l_cutoff_1 = l_peak
        integrand_1 = integrand*np.exp(-l_list*(l_list+1.)/l_cutoff_1**2.)

        # .. Third interand: damped after first peak + 10^3
        l_cutoff_2 = 1e9
        #l_cutoff_2 = l_peak + del_l
        #l_cutoff_2 = l_peak + (l_min[0]-l_peak)/2

        integrand_2 = integrand*np.exp(-l_list*(l_list+1.)/l_cutoff_2**2.)

        # Now, actually compute xi_F_plus using the two integrand cutoffs above

        # .. First, convert the list of angles to radians
        theta_list_rad = theta_list.to(u.rad).value

        # .. Cutoff 1 and 2:
        # .. .. Get xi_F_plus for each angular separation in the list of angles
        xi_F_plus_list_1 = []
        xi_F_plus_list_2 = []
        for theta in theta_list_rad:
            print(theta)
            # Temporary l_list
            l_list = np.logspace(np.log10(np.min(self.l_list)), np.log10(np.max(self.l_list)), 1e6) 
            # Get temporary integrand of xi_F_plus
            integrand = l_list * self.J0(l_list*theta)*self.P_F(l_list)
            # Get the extrema of the integrand
            l_max_indices = argrelextrema(integrand, np.greater)[0]
            l_max = np.array(([l_list[i] for i in l_max_indices]))
            l_min_indices = argrelextrema(integrand, np.less)[0]
            l_min = np.array(([l_list[i] for i in l_min_indices]))
            # Decrease the resolution of the l_list
            l_list = np.logspace(np.log10(np.min(self.l_list)), np.log10(np.max(self.l_list)), 1e5) 
            # Get combined l_list
            l_list = np.sort(np.hstack((l_list,l_min,l_max)))#np.sort(np.array(list(set(np.stack(l_list, l_max, l_min).flatten()))))
            # Get new interand
            integrand = l_list * self.J0(l_list*theta)*self.P_F(l_list)
            # Cutoff 1:
            l_cutoff_1 = l_max[0]
            integrand_1 = integrand * np.exp(-l_list*(l_list+1.)/l_cutoff_1**2.)
            # Cutoff 2:
            l_cutoff_2 = 1e9
            #l_cutoff_2 = l_cutoff_1 + del_l
            #l_cutoff_2 = l_peak + (l_min[0]-l_peak)/2
            integrand_2 = integrand * np.exp(-l_list*(l_list+1.)/l_cutoff_2**2.)
            # Now we can integrate.  We use simpson's rule for fast integration 
            integral_1 = integrate.simps(integrand_1, l_list, axis=-1)
            integral_2 = integrate.simps(integrand_2, l_list, axis=-1)
            # xi_F_plus = 1/2pi times the integral.
            xi_F_p_1 = (1/(2*np.pi))*integral_1
            xi_F_p_2 = (1/(2*np.pi))*integral_2
            # Append lists
            xi_F_plus_list_1.append(xi_F_p_1)
            xi_F_plus_list_2.append(xi_F_p_2)

        # Make plot
        f, ax = plt.subplots(2, 2, figsize=(15./(1.5), 15./(1.5*2)))
        ax[0][0].set_ylabel(r'$\ell J_0(\ell\theta) P_{\mathcal{F}}(\ell);\,\,\theta=10\,\,{\rm arcsec}$')
        ax[0][0].semilogx(l_list, integrand, color='black', zorder=2)

        ax[0][0].semilogx(l_list, integrand, color='black', zorder=2)


        ax[0][0].axvline(l_cutoff_1, linestyle='--', color='blue', zorder=1)
        ax[0][0].axvline(l_cutoff_2, linestyle='--', color='red', zorder=1)
        ax[1][0].set_ylabel(r'$\ell J_0(\ell\theta) P_{\mathcal{F}}(\ell) e^{-\ell(\ell+1)/\ell_{\rm cutoff}^2}$')
        ax[1][0].semilogx(l_list, integrand_1, color='blue', zorder=1)
        ax[1][0].semilogx(l_list, integrand_2, color='red', zorder=2)

        ax[1][0].set_xlabel(r'$\ell$')

        ax[0][1].loglog(theta_list.value, xi_F_plus_list_1, color='blue') 
        ax[0][1].loglog(theta_list.value, xi_F_plus_list_2, color='red') 
        ax[0][1].set_ylabel(r'$\xi_{\mathcal{F}+}(\theta)$')

        ax[1][1].semilogx(theta_list.value, 100.*(np.array(xi_F_plus_list_1)-np.array(xi_F_plus_list_2))/np.array(xi_F_plus_list_1), color = 'black')
        ax[1][1].set_ylabel(r'${\rm rel.\,\,dev.\,\,[\%]}$')

        ax[1][1].set_xlabel(r'$\theta\,\,{\rm [arcsec]}$')

        plt.tight_layout()
        plt.savefig('cosmic_flexion_xi_F_plus_conv_demo.pdf', format='pdf')
        #plt.show()






































        
