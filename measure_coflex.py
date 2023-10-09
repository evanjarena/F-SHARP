import numpy as np
import sys,os
import pandas as pd
from astropy import units as u

"""
Script: Measure the cosmic flexion and shear-flexion two-point correlation functions from a dataset.

Author: Evan J. Arena

Description:
.. We want to compute the two-point statistics on a dataset given 
     galaxy position, size, ellipticity, and flexion
.. We will compute the following statistics:
    (i).   All cosmic flexion-flexion correlators:
            xi_FF_plus, xi_FF_minus,
            xi_GG_plus, xi_GG_minus,
            xi_FG_plus, xi_FG_minus, xi_GF_plus, xi_GF_minus
    (ii).  All cosmic shear-flexion correlators:
            xi_gamF_plus, xi_gamF_minus, xi_Fgam_plus, xi_Fgam_minus
            xi_Ggam_plus, xi_Ggam_minus, xi_gamG_plus, xi_gamG_minus, 
    (iii). All autovariances of each estimator
"""

class MeasureCF2P:

    def __init__(self, survey, bin_combo, RA, DEC, F1, F2, G1, G2, a, eps1, eps2, w):
        """
        We take as an input a galaxy "catalogue" containing the following parameters:
         .. survey: a string denoting the name of the cosmological survey, 
              e.g. 'DES'
         .. bin_combo: a string denoting the tomographic bin combination
              e.g. '11' for bin combination (1,1), 'nontom' for nontomographic analysis
         .. RA: Array of the right-ascention of each galaxy, in arcseconds
         .. DEC: Array of the declination of each galaxy, in arcseconds
         .. F1: Array of the F-flexion 1-component of each galaxy, in [arcsec]^-1
         .. F2: Array of the F-flexion 2-component  of each galaxy, in [arcsec]^-1
         .. G1: Array of the G-flexion 1-component of each galaxy, in [arcsec]^-1
         .. G2: Array of the G-flexion 2-component of each galaxy, in [arcsec]^-1
         .. a: Array of the sizes of each galaxy, in arcseconds.  Size is defined
              in terms of the quandrupole image moments: 
                a = sqrt(|Q11 + Q22|)
         .. eps1: Array of the ellipticity 1-component of each galaxy*.
         .. eps2: Array of the ellipticity 2-component of each galaxy*.
         .. w: Array of weights for each galaxy.

         * Note that ellipticity is defined here as (a-b)/(a+b)e^2*i*phi, where a and b
           are the semi-major and semi-minor axes. Do not confuse the semi-major axis here
           with the image size.
        """
        self.survey = str(survey)
        self.bin_combo = str(bin_combo)
        self.x_list = RA
        self.y_list = DEC
        self.F1_list = F1
        self.F2_list = F2
        self.G1_list = G1
        self.G2_list = G2
        self.a_list = a
        self.eps1_list = eps1
        self.eps2_list = eps2
        self.w_list = w
        self.Ngals = len(self.w_list)

    def measureTwoPoint(self):
        """
        Measure all available two-point correlation functions.
        """
        # First, calculate the scatter in intrinsic flexion and ellipticity
        sigma_aF, sigma_aG, sigma_eps = self.get_intrinsic_scatter()
        print('Intrinsic scatter:')
        print('sigma_aF, sigma_aG, sigma_eps =', sigma_aF, sigma_aG, sigma_eps)
        self.sigma_aF = sigma_aF
        self.sigma_aG = sigma_aG
        self.sigma_eps = sigma_eps
        # Perform mean-subtraction to remove additive biases     
        print('Means of lensing fields')
        print('<F1>, <F2> =', np.mean(self.F1_list), np.mean(self.F2_list))
        print('<G1>, <G2> =', np.mean(self.G1_list), np.mean(self.G2_list))
        print('<eps1>, <eps2> =', np.mean(self.eps1_list), np.mean(self.eps2_list))
        # Subtract means
        self.F1_list -= np.mean(self.F1_list)
        self.F2_list -= np.mean(self.F2_list)
        self.G1_list -= np.mean(self.G1_list)
        self.G2_list -= np.mean(self.G2_list)
        self.eps1_list -= np.mean(self.eps1_list)
        self.eps2_list -= np.mean(self.eps2_list)
        print('Means of mean-subtracted lensing fields')
        print('<F1>, <F2> =', np.mean(self.F1_list), np.mean(self.F2_list))
        print('<G1>, <G2> =', np.mean(self.G1_list), np.mean(self.G2_list))
        print('<eps1>, <eps2> =', np.mean(self.eps1_list), np.mean(self.eps2_list))
        # Get flexion-flexion correlation functions first. 
        # .. Get angular separations and bins
        theta_flexflex_list, flexflex_bins = self.theta_flexflex_bin()
        # .. Get weights and two-point arrays in each bin
        # .. .. flexflex_bins = array(wp_bins, xi_FF_plus_bins, ...)
        self.get_binned_two_point_flexflex(theta_flexflex_list, flexflex_bins)
        # .. Get two-point correlation functions and autovariances
        
        # Shear-flexion correlations
        # .. Get angular separations and bins
        theta_shearflex_list, shearflex_bins = self.theta_shearflex_bin()
        # .. Get weights and two-point arrays in each bin
        # .. .. flexflex_bins = array(wp_bins, xi_FF_plus_bins, ...)
        self.get_binned_two_point_shearflex(theta_shearflex_list, shearflex_bins)
        # .. Get two-point correlation functions
        
    def theta_flexflex_bin(self, theta_min=1, theta_max=100, N_theta=10):
        """
        List of theta values for real-space cosmic flexion correlation functions
        Input angle values are in untis of arcseconds
        """
        theta_min = np.log10(theta_min)
        theta_max = np.log10(theta_max)
        theta_list = np.logspace(theta_min,theta_max,N_theta)
        dtheta = np.log10(theta_list[1])-np.log10(theta_list[0])
        bin_low_list = 10**(np.log10(theta_list)-dtheta/2)
        bin_high_list = 10**(np.log10(theta_list)+dtheta/2)
        bins = np.append(bin_low_list,bin_high_list[-1])
        return theta_list, bins

    def theta_shearflex_bin(self, theta_min=1/60, theta_max=10, N_theta=15):
        """
        List of theta values for real-space cosmic shear-flexion correlation functions
        Input angle values are in untis of arcminutes
        """
        theta_min = np.log10(theta_min)
        theta_max = np.log10(theta_max)
        theta_list = np.logspace(theta_min,theta_max,N_theta)
        dtheta = np.log10(theta_list[1])-np.log10(theta_list[0])
        bin_low_list = 10**(np.log10(theta_list)-dtheta/2)
        bin_high_list = 10**(np.log10(theta_list)+dtheta/2)
        bins = np.append(bin_low_list,bin_high_list[-1])
        return theta_list, bins

    def get_binned_two_point_flexflex(self, theta_list, bins):
        """
        Calculate the following quantities for each galaxy pair (i,j):
          1. The product of galaxy weights:
               w_p = w_i*w_j
          2. The two-point correlations, e.g.
               xi_FF_p/m = (F1_rot_i*F1_rot_j +/- F2_rot_i*F2_rot_j)
        and then separate each galaxy pair into angular separation bins
        defined by the function self.theta_bin().

        Rather than perform this calculation for every single galaxy pair, i.e.
          >>> for i in range(N_bins):
          >>> .. for j in range(N_bins):
        which is an O(N^2) operation, we turn this into an ~O(N) operation by
        creating square grid cells with widths equal to the largest angular 
        separation we consider: np.max(theta_list).
        """
        # Get the total number of bins
        N_bins = len(theta_list)
        # Define the arrays for two-point calculation
        # .. Number of pairs (product of weights)
        Np = np.zeros(N_bins)
        # .. Auto-correlations
        xi_FF_plus_bins = np.zeros(N_bins)
        xi_FF_minus_bins = np.zeros(N_bins)
        xi_FF_plus_autoVar_bins = np.zeros(N_bins)
        xi_FF_minus_autoVar_bins = np.zeros(N_bins)
        xi_GG_plus_bins = np.zeros(N_bins)
        xi_GG_minus_bins = np.zeros(N_bins)
        xi_GG_plus_autoVar_bins = np.zeros(N_bins)
        xi_GG_minus_autoVar_bins = np.zeros(N_bins)
        # .. Cross-correlations
        xi_FG_plus_bins = np.zeros(N_bins)
        xi_FG_minus_bins = np.zeros(N_bins)
        xi_FG_plus_autoVar_bins = np.zeros(N_bins)
        xi_FG_minus_autoVar_bins = np.zeros(N_bins)
        xi_GF_plus_bins = np.zeros(N_bins)
        xi_GF_minus_bins = np.zeros(N_bins)
        xi_GF_plus_autoVar_bins = np.zeros(N_bins)
        xi_GF_minus_autoVar_bins = np.zeros(N_bins)
        # .. B-mode correlations
        xi_FF_cross1_bins = np.zeros(N_bins)
        xi_FF_cross2_bins = np.zeros(N_bins)
        xi_GG_cross1_bins = np.zeros(N_bins)
        xi_GG_cross2_bins = np.zeros(N_bins)
        xi_FG_cross1_bins = np.zeros(N_bins)
        xi_FG_cross2_bins = np.zeros(N_bins)
        xi_GF_cross1_bins = np.zeros(N_bins)
        xi_GF_cross2_bins = np.zeros(N_bins)

        # Get the width of each grid cell:
        dg = np.max(bins)
        # Let (Gx,Gy) denote the grid pairs.  We can assign each galaxy, k, to a
        # grid in the following way:
        gx_list = np.zeros(self.Ngals)
        gy_list = np.zeros(self.Ngals)
        for k in range(self.Ngals):
            gx = int((self.x_list[k]-self.x_list[0])/dg)
            gy = int((self.y_list[k]-self.y_list[0])/dg)
            gx_list[k] = gx
            gy_list[k] = gy

        # Next, we want to loop through every galaxy, i.  We want to calculate 
        # the separation and the two-point statistics, between i and all other 
        # galaxies, j, that lie in either the same grid cell as i or a grid cell
        # adjacent to it. Now, unless we account for the fact that there are 
        # multiple galaxies in each grid cell, the separation between galaxy
        # i and i+1 will be calculated on both the 0th and 1st iterations.  
        # Therefore, we should create a running list containing galaxies already
        # looped through and exclude them from the next iteration.  We can do this
        # simply by requiring j > i.  This is fine because for each iteration (i),
        # all of (i)'s pairs are identified.
        
        # Create list of galaxy j:
        j_list = np.arange(0, self.Ngals)

        for i in range(self.Ngals):

            # Get galaxy pairs {j} associated with galaxy i
            id = np.where((j_list > i) &
                          (gx_list <= gx_list[i]+1) & (gx_list >= gx_list[i]-1) &
                          (gy_list <= gy_list[i]+1) & (gy_list >= gy_list[i]-1))
            # Positions of each galaxy j
            x_j_list = self.x_list[id]
            y_j_list = self.y_list[id]
            # Get total number of galaxy js
            N_j = len(x_j_list)
            # Sizes
            a_j_list = self.a_list[id]
            # Weights
            w_j_list = self.w_list[id]
            # Flexions
            F1_j_list = self.F1_list[id]
            F2_j_list = self.F2_list[id]
            G1_j_list = self.G1_list[id]
            G2_j_list = self.G2_list[id]

            # Calculate two-point for each pair (i,j)
            for j in range(N_j):
                # Get separation between (i,j)
                theta_ij = np.sqrt((x_j_list[j]-self.x_list[i])**2.+(y_j_list[j]-self.y_list[i])**2.)
                if theta_ij >= np.max(bins):
                    continue
                # Get polar angle between (i,j)
                varPhi_ij = np.arctan2((y_j_list[j]-self.y_list[i]), (x_j_list[j]-self.x_list[i]))
                # Get trig functions associated with polar angle
                cos_varPhi_ij = np.cos(varPhi_ij)
                sin_varPhi_ij = np.sin(varPhi_ij)
                cos_3varPhi_ij = np.cos(3*varPhi_ij)
                sin_3varPhi_ij = np.sin(3*varPhi_ij)
                # Calculate rotated flexions for pair (i,j)
                F1_rot_i = -self.F1_list[i]*cos_varPhi_ij - self.F2_list[i]*sin_varPhi_ij
                F1_rot_j = -F1_j_list[j]*cos_varPhi_ij - F2_j_list[j]*sin_varPhi_ij
                F2_rot_i = -self.F2_list[i]*cos_varPhi_ij + self.F1_list[i]*sin_varPhi_ij
                F2_rot_j = -F2_j_list[j]*cos_varPhi_ij + F1_j_list[j]*sin_varPhi_ij
                G1_rot_i = self.G1_list[i]*cos_3varPhi_ij + self.G2_list[i]*sin_3varPhi_ij 
                G1_rot_j = G1_j_list[j]*cos_3varPhi_ij + G2_j_list[j]*sin_3varPhi_ij
                G2_rot_i = self.G2_list[i]*cos_3varPhi_ij - self.G1_list[i]*sin_3varPhi_ij
                G2_rot_j = G2_j_list[j]*cos_3varPhi_ij - G1_j_list[j]*sin_3varPhi_ij
                # Weight for each pair
                wp_ij = self.w_list[i]*w_j_list[j]
                # Two-points for each pair
                xi_FF_p_ij = wp_ij*(F1_rot_i*F1_rot_j + F2_rot_i*F2_rot_j)
                xi_FF_m_ij = wp_ij*(F1_rot_i*F1_rot_j - F2_rot_i*F2_rot_j)
                xi_GG_p_ij = wp_ij*(G1_rot_i*G1_rot_j + G2_rot_i*G2_rot_j)
                xi_GG_m_ij = wp_ij*(G1_rot_i*G1_rot_j - G2_rot_i*G2_rot_j)
                xi_FG_p_ij = wp_ij*(F1_rot_i*G1_rot_j + F2_rot_i*G2_rot_j)
                xi_FG_m_ij = wp_ij*(F1_rot_i*G1_rot_j - F2_rot_i*G2_rot_j)
                xi_GF_p_ij = wp_ij*(G1_rot_i*F1_rot_j + G2_rot_i*F2_rot_j)
                xi_GF_m_ij = wp_ij*(G1_rot_i*F1_rot_j - G2_rot_i*F2_rot_j)
                # Autovar for each pair
                xi_FF_p_aV_ij = (wp_ij/(self.a_list[i]*a_j_list[j]))**2.
                #xi_FF_m_aV_ij = xi_FF_p_aV_ij
                #xi_GG_p_aV_ij = xi_FF_p_aV_ij
                #xi_GG_m_aV_ij = xi_FF_p_aV_ij
                #xi_FG_p_aV_ij = xi_FF_p_aV_ij
                #xi_FG_m_aV_ij = xi_FF_p_aV_ij
                #xi_GF_p_aV_ij = xi_FF_p_aV_ij
                #xi_GF_m_aV_ij = xi_FF_p_aV_ij
                # B-mode correlations for each pair
                xi_FF_c1_ij = wp_ij*F1_rot_i*F2_rot_j
                xi_FF_c2_ij = wp_ij*F2_rot_i*F1_rot_j
                xi_GG_c1_ij = wp_ij*G1_rot_i*G2_rot_j
                xi_GG_c2_ij = wp_ij*G2_rot_i*G1_rot_j
                xi_FG_c1_ij = wp_ij*F1_rot_i*G2_rot_j
                xi_FG_c2_ij = wp_ij*F2_rot_i*G1_rot_j
                xi_GF_c1_ij = wp_ij*G1_rot_i*F2_rot_j
                xi_GF_c2_ij = wp_ij*G2_rot_i*F1_rot_j

                # Get the bin for each galaxy pair (i,j).  It is simplest to use
                # np.digitize(theta_ij, bins).  This returns the bin number that 
                # theta_ij belongs to (digitize indexes at 1). If digitize returns 0,
                # theta_ij is smaller than the smallest bin.  If digitize returns the 
                # number = N_bins + 1, then theta_ij is larger than the largest bin. So
                bin_ij = np.digitize(theta_ij, bins, right=True)
                if (bin_ij > 0) & (bin_ij < N_bins + 1):
                    bin_index = bin_ij-1
                    
                    Np[bin_index] += wp_ij
                    
                    xi_FF_plus_bins[bin_index] += xi_FF_p_ij
                    xi_FF_minus_bins[bin_index] += xi_FF_m_ij
                    xi_GG_plus_bins[bin_index] += xi_GG_p_ij
                    xi_GG_minus_bins[bin_index] += xi_GG_m_ij
                    xi_FG_plus_bins[bin_index] += xi_FG_p_ij
                    xi_FG_minus_bins[bin_index] += xi_FG_m_ij
                    xi_GF_plus_bins[bin_index] += xi_GF_p_ij
                    xi_GF_minus_bins[bin_index] += xi_GF_m_ij

                    xi_FF_plus_autoVar_bins[bin_index] += xi_FF_p_aV_ij
                    #xi_FF_minus_autoVar_bins[bin_index] += xi_FF_m_aV_ij
                    #xi_GG_plus_autoVar_bins[bin_index] += xi_GG_p_aV_ij
                    #xi_GG_minus_autoVar_bins[bin_index] += xi_GG_m_aV_ij
                    #xi_FG_plus_autoVar_bins[bin_index] += xi_FG_p_aV_ij
                    #xi_FG_minus_autoVar_bins[bin_index] += xi_FG_m_aV_ij
                    #xi_GF_plus_autoVar_bins[bin_index] += xi_GF_p_aV_ij
                    #xi_GF_minus_autoVar_bins[bin_index] += xi_GF_m_aV_ij
                    
                    xi_FF_cross1_bins[bin_index] += xi_FF_c1_ij
                    xi_FF_cross2_bins[bin_index] += xi_FF_c2_ij
                    xi_GG_cross1_bins[bin_index] += xi_GG_c1_ij
                    xi_GG_cross2_bins[bin_index] += xi_GG_c2_ij
                    xi_FG_cross1_bins[bin_index] += xi_FG_c1_ij
                    xi_FG_cross2_bins[bin_index] += xi_FG_c2_ij
                    xi_GG_cross1_bins[bin_index] += xi_GG_c1_ij
                    xi_GG_cross2_bins[bin_index] += xi_GG_c2_ij

        xi_FF_plus = xi_FF_plus_bins/Np
        xi_FF_minus = xi_FF_minus_bins/Np
        xi_GG_plus = xi_GG_plus_bins/Np
        xi_GG_minus = xi_GG_minus_bins/Np
        xi_FG_plus = xi_FG_plus_bins/Np
        xi_FG_minus = xi_FG_minus_bins/Np
        xi_GF_plus = xi_GF_plus_bins/Np
        xi_GF_minus = xi_GF_minus_bins/Np         
        
        #xi_FF_plus_autoVar = xi_FF_plus_autoVar_bins*(self.sigma_aF**4./(2*Np**2.))
        #xi_FF_minus_autoVar = xi_FF_minus_autoVar_bins*(self.sigma_aF**4./(2*Np**2.))
        #xi_GG_plus_autoVar = xi_GG_plus_autoVar_bins*(self.sigma_aG**4./(2*Np**2.))
        #xi_GG_minus_autoVar = xi_GG_minus_autoVar_bins*(self.sigma_aG**4./(2*Np**2.))
        #xi_FG_plus_autoVar = xi_FG_plus_autoVar_bins*(self.sigma_aF**2.*self.sigma_aG**2./(2*Np**2.))
        #xi_FG_minus_autoVar = xi_FG_minus_autoVar_bins*(self.sigma_aF**2.*self.sigma_aG**2./(2*Np**2.))
        #xi_GF_plus_autoVar = xi_GF_plus_autoVar_bins*(self.sigma_aF**2.*self.sigma_aG**2./(2*Np**2.))
        #xi_GF_minus_autoVar = xi_GF_minus_autoVar_bins*(self.sigma_aF**2.*self.sigma_aG**2./(2*Np**2.))

        xi_FF_plus_autoVar = xi_FF_plus_autoVar_bins*(self.sigma_aF**4./(2*Np**2.))
        xi_FF_minus_autoVar = xi_FF_plus_autoVar_bins*(self.sigma_aF**4./(2*Np**2.))
        xi_GG_plus_autoVar = xi_FF_plus_autoVar_bins*(self.sigma_aG**4./(2*Np**2.))
        xi_GG_minus_autoVar = xi_FF_plus_autoVar_bins*(self.sigma_aG**4./(2*Np**2.))
        xi_FG_plus_autoVar = xi_FF_plus_autoVar_bins*(self.sigma_aF**2.*self.sigma_aG**2./(2*Np**2.))
        xi_FG_minus_autoVar = xi_FF_plus_autoVar_bins*(self.sigma_aF**2.*self.sigma_aG**2./(2*Np**2.))
        xi_GF_plus_autoVar = xi_FF_plus_autoVar_bins*(self.sigma_aF**2.*self.sigma_aG**2./(2*Np**2.))
        xi_GF_minus_autoVar = xi_FF_plus_autoVar_bins*(self.sigma_aF**2.*self.sigma_aG**2./(2*Np**2.))

        xi_FF_cross1 = xi_FF_cross1_bins/Np
        xi_FF_cross2 = xi_FF_cross2_bins/Np
        xi_GG_cross1 = xi_GG_cross1_bins/Np
        xi_GG_cross2 = xi_GG_cross2_bins/Np
        xi_FG_cross1 = xi_FG_cross1_bins/Np
        xi_FG_cross2 = xi_FG_cross2_bins/Np
        xi_GF_cross1 = xi_GF_cross1_bins/Np
        xi_GF_cross2 = xi_GF_cross2_bins/Np

        # Convert flexions from 1/arcsec^2 to 1/rad^2
        xi_FF_plus = (xi_FF_plus/u.arcsec**2.).to(1/u.rad**2.).value
        xi_FF_minus = (xi_FF_minus/u.arcsec**2.).to(1/u.rad**2.).value
        xi_GG_plus = (xi_GG_plus/u.arcsec**2.).to(1/u.rad**2.).value
        xi_GG_minus = (xi_GG_minus/u.arcsec**2.).to(1/u.rad**2.).value
        xi_FG_plus = (xi_FG_plus/u.arcsec**2.).to(1/u.rad**2.).value
        xi_FG_minus = (xi_FG_minus/u.arcsec**2.).to(1/u.rad**2.).value
        xi_GF_plus = (xi_GF_plus/u.arcsec**2.).to(1/u.rad**2.).value
        xi_GF_minus = (xi_GF_minus/u.arcsec**2.).to(1/u.rad**2.).value

        xi_FF_cross1 = (xi_FF_cross1/u.arcsec**2.).to(1/u.rad**2.).value
        xi_FF_cross2 = (xi_FF_cross2/u.arcsec**2.).to(1/u.rad**2.).value
        xi_GG_cross1 = (xi_GG_cross1/u.arcsec**2.).to(1/u.rad**2.).value
        xi_GG_cross2 = (xi_GG_cross2/u.arcsec**2.).to(1/u.rad**2.).value
        xi_FG_cross1 = (xi_FG_cross1/u.arcsec**2.).to(1/u.rad**2.).value
        xi_FG_cross2 = (xi_FG_cross2/u.arcsec**2.).to(1/u.rad**2.).value
        xi_GG_cross1 = (xi_GG_cross1/u.arcsec**2.).to(1/u.rad**2.).value
        xi_GG_cross2 = (xi_GG_cross2/u.arcsec**2.).to(1/u.rad**2.).value

        xi_FF_plus_autoVar = (xi_FF_plus_autoVar/u.arcsec**4.).to(1/u.rad**4.).value
        xi_FF_minus_autoVar = (xi_FF_minus_autoVar/u.arcsec**4.).to(1/u.rad**4.).value
        xi_GG_plus_autoVar = (xi_GG_plus_autoVar/u.arcsec**4.).to(1/u.rad**4.).value
        xi_GG_minus_autoVar = (xi_GG_minus_autoVar/u.arcsec**4.).to(1/u.rad**4.).value
        xi_FG_plus_autoVar = (xi_FG_plus_autoVar/u.arcsec**4.).to(1/u.rad**4.).value
        xi_FG_minus_autoVar = (xi_FG_minus_autoVar/u.arcsec**4.).to(1/u.rad**4.).value
        xi_GF_plus_autoVar = (xi_GF_plus_autoVar/u.arcsec**4.).to(1/u.rad**4.).value
        xi_GF_minus_autoVar = (xi_GF_minus_autoVar/u.arcsec**4.).to(1/u.rad**4.).value

        # .. Export flexion-flexion correlation functions to .pkl file
        col_list = ['theta', 'Np', 
                    'xi_FF_plus', 'xi_FF_plus_autoVar', 'xi_FF_minus', 'xi_FF_minus_autoVar', 
                    'xi_GG_plus', 'xi_GG_plus_autoVar', 'xi_GG_minus', 'xi_GG_minus_autoVar', 
                    'xi_FG_plus', 'xi_FG_plus_autoVar', 'xi_FG_minus', 'xi_FG_minus_autoVar', 
                    'xi_GF_plus', 'xi_GF_plus_autoVar', 'xi_GF_minus', 'xi_GF_minus_autoVar', 
                    'xi_FF_cross1', 'xi_FF_cross2',
                    'xi_GG_cross1', 'xi_GG_cross2',
                    'xi_FG_cross1', 'xi_FG_cross2',
                    'xi_GF_cross1', 'xi_GF_cross2']
        arrs = [theta_list, Np, 
                xi_FF_plus, xi_FF_plus_autoVar, xi_FF_minus, xi_FF_minus_autoVar, 
                xi_GG_plus, xi_GG_plus_autoVar, xi_GG_minus, xi_GG_minus_autoVar, 
                xi_FG_plus, xi_FG_plus_autoVar, xi_FG_minus, xi_FG_minus_autoVar, 
                xi_GF_plus, xi_GF_plus_autoVar, xi_GF_minus, xi_GF_minus_autoVar, 
                xi_FF_cross1, xi_FF_cross2,
                xi_GG_cross1, xi_GG_cross2,
                xi_FG_cross1, xi_FG_cross2,
                xi_GF_cross1, xi_GF_cross2]
        dat = {i:arrs[j] for i,j in zip(col_list, range(len(col_list)))}
        out_frame = pd.DataFrame(data = dat, columns = col_list)
        #out_frame.to_pickle(self.survey+'/'+self.survey+'_Measure/flexion-flexion_two_point_measured_'+self.survey+'_bin_combo_'+self.bin_combo+'.pkl')
        out_frame.to_csv('flexion-flexion_two_point_measured_'+self.survey+'_bin_combo_'+self.bin_combo+'.csv')

    def get_binned_two_point_shearflex(self, theta_list, bins):
        """
        Calculate the following quantities for each galaxy pair (i,j):
          1. The product of galaxy weights:
               w_p = w_i*w_j
          2. The two-point correlations, e.g.
               xi_gamF_p/m = (gam1_rot_i*F1_rot_j +/- gam2_rot_i*F2_rot_j)
        and then separate each galaxy pair into angular separation bins
        defined by the function self.theta_bin().

        Rather than perform this calculation for every single galaxy pair, i.e.
          >>> for i in range(N_bins):
          >>> .. for j in range(N_bins):
        which is an O(N^2) operation, we turn this into an ~O(N) operation by
        creating square grid cells with widths equal to the largest angular 
        separation we consider: np.max(theta_list).
        """
        # First, get positions in arcminutes
        x_list = self.x_list/60
        y_list = self.y_list/60

        # Get the total number of bins
        N_bins = len(theta_list)
        # Define the arrays for two-point calculation
        # .. Number of pairs (product of weights)
        Np = np.zeros(N_bins)
        # .. Two-point functions
        xi_gamF_plus_bins = np.zeros(N_bins)
        xi_gamF_minus_bins = np.zeros(N_bins)
        xi_Fgam_plus_bins = np.zeros(N_bins)
        xi_Fgam_minus_bins = np.zeros(N_bins)
        xi_Ggam_plus_bins = np.zeros(N_bins)
        xi_Ggam_minus_bins = np.zeros(N_bins)
        xi_gamG_plus_bins = np.zeros(N_bins)
        xi_gamG_minus_bins = np.zeros(N_bins)
        # .. Autovariance
        xi_gamF_plus_autoVar_bins = np.zeros(N_bins)
        xi_gamF_minus_autoVar_bins = np.zeros(N_bins)
        xi_Fgam_plus_autoVar_bins = np.zeros(N_bins)
        xi_Fgam_minus_autoVar_bins = np.zeros(N_bins)
        xi_Ggam_plus_autoVar_bins = np.zeros(N_bins)
        xi_Ggam_minus_autoVar_bins = np.zeros(N_bins)
        xi_gamG_plus_autoVar_bins = np.zeros(N_bins)
        xi_gamG_minus_autoVar_bins = np.zeros(N_bins)
        # B-mode correlations
        xi_gamF_cross1 = np.zeros(N_bins)
        xi_gamF_cross2 = np.zeros(N_bins)
        xi_Fgam_cross1 = np.zeros(N_bins)
        xi_Fgam_cross2 = np.zeros(N_bins)
        xi_Ggam_cross1 = np.zeros(N_bins)
        xi_Ggam_cross2 = np.zeros(N_bins)
        xi_gamG_cross1 = np.zeros(N_bins)
        xi_gamG_cross2 = np.zeros(N_bins)

        # Get the width of each grid cell:
        dg = np.max(bins)
        # Let (Gx,Gy) denote the grid pairs.  We can assign each galaxy, k, to a
        # grid in the following way:
        gx_list = np.zeros(self.Ngals)
        gy_list = np.zeros(self.Ngals)
        for k in range(self.Ngals):
            gx = int((x_list[k]-x_list[0])/dg)
            gy = int((y_list[k]-y_list[0])/dg)
            gx_list[k] = gx
            gy_list[k] = gy

        # Next, we want to loop through every galaxy, i.  We want to calculate 
        # the separation and the two-point statistics, between i and all other 
        # galaxies, j, that lie in either the same grid cell as i or a grid cell
        # adjacent to it. Now, unless we account for the fact that there are 
        # multiple galaxies in each grid cell, the separation between galaxy
        # i and i+1 will be calculated on both the 0th and 1st iterations.  
        # Therefore, we should create a running list containing galaxies already
        # looped through and exclude them from the next iteration.  We can do this
        # simply by requiring j > i.  This is fine because for each iteration (i),
        # all of (i)'s pairs are identified.
        
        # Create list of galaxy j:
        j_list = np.arange(0, self.Ngals)

        for i in range(self.Ngals):

            # Get galaxy pairs {j} associated with galaxy i
            id = np.where((j_list > i) &
                          (gx_list <= gx_list[i]+1) & (gx_list >= gx_list[i]-1) &
                          (gy_list <= gy_list[i]+1) & (gy_list >= gy_list[i]-1))
            # Positions of each galaxy j
            x_j_list = x_list[id]
            y_j_list = y_list[id]
            # Get total number of galaxy js
            N_j = len(x_j_list)
            # Sizes
            a_j_list = self.a_list[id]
            # Weights
            w_j_list = self.w_list[id]
            # Flexions
            F1_j_list = self.F1_list[id]
            F2_j_list = self.F2_list[id]
            G1_j_list = self.G1_list[id]
            G2_j_list = self.G2_list[id]
            # Ellipticities
            eps1_j_list = self.eps1_list[id]
            eps2_j_list = self.eps2_list[id]           
            
            # Calculate two-point for each pair (i,j)
            for j in range(N_j):

                # Get separation between (i,j)
                theta_ij = np.sqrt((x_j_list[j]-self.x_list[i])**2.+(y_j_list[j]-self.y_list[i])**2.)
                if theta_ij >= np.max(bins):
                    continue
                # Get polar angle between (i,j)
                varPhi_ij = np.arctan2((y_j_list[j]-self.y_list[i]), (x_j_list[j]-self.x_list[i]))
                # Get trig functions associated with polar angle 
                cos_varPhi_ij = np.cos(varPhi_ij)
                sin_varPhi_ij = np.sin(varPhi_ij)
                cos_2varPhi_ij = np.cos(2*varPhi_ij)
                sin_2varPhi_ij = np.sin(2*varPhi_ij)
                cos_3varPhi_ij = np.cos(3*varPhi_ij)
                sin_3varPhi_ij = np.sin(3*varPhi_ij)
                # Get rotated flexions and ellipticities for (i,j)
                F1_rot_i = -self.F1_list[i]*cos_varPhi_ij - self.F2_list[i]*sin_varPhi_ij
                F1_rot_j = -F1_j_list[j]*cos_varPhi_ij - F2_j_list[j]*sin_varPhi_ij
                F2_rot_i = -self.F2_list[i]*cos_varPhi_ij + self.F1_list[i]*sin_varPhi_ij 
                F2_rot_j = -F2_j_list[j]*cos_varPhi_ij + F1_j_list[j]*sin_varPhi_ij 
                G1_rot_i = self.G1_list[i]*cos_3varPhi_ij + self.G2_list[i]*sin_3varPhi_ij 
                G1_rot_j = G1_j_list[j]*cos_3varPhi_ij + G2_j_list[j]*sin_3varPhi_ij
                G2_rot_i = self.G2_list[i]*cos_3varPhi_ij - self.G1_list[i]*sin_3varPhi_ij
                G2_rot_j = G2_j_list[j]*cos_3varPhi_ij - G1_j_list[j]*sin_3varPhi_ij
                eps1_rot_i = -self.eps1_list[i]*cos_2varPhi_ij - self.eps2_list[i]*sin_2varPhi_ij
                eps1_rot_j = -eps1_j_list[j]*cos_2varPhi_ij - eps2_j_list[j]*sin_2varPhi_ij
                eps2_rot_i = -self.eps2_list[i]*cos_2varPhi_ij + self.eps1_list[i]*sin_2varPhi_ij 
                eps2_rot_j = -eps2_j_list[j]*cos_2varPhi_ij + eps1_j_list[j]*sin_2varPhi_ij

                # Weight for each pair
                wp_ij = self.w_list[i]*w_j_list[j]
                # Two-points for each pair
                xi_epsF_p_ij = wp_ij*(eps1_rot_i*F1_rot_j + eps2_rot_i*F2_rot_j)
                xi_epsF_m_ij = wp_ij*(eps1_rot_i*F1_rot_j - eps2_rot_i*F2_rot_j)
                xi_Feps_p_ij = wp_ij*(F1_rot_i*eps1_rot_j + F2_rot_i*eps2_rot_j)
                xi_Feps_m_ij = wp_ij*(F1_rot_i*eps1_rot_j - F2_rot_i*eps2_rot_j)
                xi_Geps_p_ij = wp_ij*(G1_rot_i*eps1_rot_j + G2_rot_i*eps2_rot_j)
                xi_Geps_m_ij = wp_ij*(G1_rot_i*eps1_rot_j - G2_rot_i*eps2_rot_j)
                xi_epsG_p_ij = wp_ij*(eps1_rot_i*G1_rot_j + eps2_rot_i*G2_rot_j)
                xi_epsG_m_ij = wp_ij*(eps1_rot_i*G1_rot_j - eps2_rot_i*G2_rot_j)
                # Autovar for each pair
                xi_epsF_p_aV_ij = (wp_ij/(a_j_list[j]))**2.
                xi_epsF_m_aV_ij = (wp_ij/(a_j_list[j]))**2.
                xi_Feps_p_aV_ij = (wp_ij/(self.a_list[i]))**2.
                xi_Feps_m_aV_ij = (wp_ij/(self.a_list[i]))**2.
                xi_Geps_p_aV_ij = (wp_ij/(self.a_list[i]))**2.
                xi_Geps_m_aV_ij = (wp_ij/(self.a_list[i]))**2.
                xi_epsG_p_aV_ij = (wp_ij/(a_j_list[j]))**2.
                xi_epsG_m_aV_ij = (wp_ij/(a_j_list[j]))**2.
                # B-modes for each pair
                xi_epsF_cross1 = wp_ij*(eps1_rot_i*F2_rot_j)
                xi_epsF_cross2 = wp_ij*(eps2_rot_i*F1_rot_j)
                xi_Feps_cross1 = wp_ij*(F1_rot_i*eps2_rot_j)
                xi_Feps_cross2 = wp_ij*(F2_rot_i*eps1_rot_j)
                xi_Geps_cross1 = wp_ij*(G1_rot_i*eps2_rot_j)
                xi_Geps_cross2 = wp_ij*(G2_rot_i*eps1_rot_j)
                xi_epsG_cross1 = wp_ij*(eps1_rot_i*G2_rot_j)
                xi_epsG_cross2 = wp_ij*(eps2_rot_i*G1_rot_j)

                # Get the bin for each galaxy pair (i,j).  It is simplest to use
                # np.digitize(theta_ij, bins).  This returns the bin number that 
                # theta_ij belongs to (digitize indexes at 1). If digitize returns 0,
                # theta_ij is smaller than the smallest bin.  If digitize returns the 
                # number = N_bins + 1, then theta_ij is larger than the largest bin. So
                bin_ij = np.digitize(theta_ij, bins, right=True)
                if (bin_ij > 0) & (bin_ij < N_bins + 1):
                    bin_index = bin_ij-1
                    Np[bin_index] += wp_ij
                    xi_gamF_plus_bins[bin_index] += xi_epsF_p_ij
                    xi_gamF_minus_bins[bin_index] += xi_epsF_m_ij
                    xi_Fgam_plus_bins[bin_index] += xi_Feps_p_ij
                    xi_Fgam_minus_bins[bin_index] += xi_Feps_m_ij
                    xi_Ggam_plus_bins[bin_index] += xi_Geps_p_ij
                    xi_Ggam_minus_bins[bin_index] += xi_Geps_m_ij
                    xi_gamG_plus_bins[bin_index] += xi_epsG_p_ij
                    xi_gamG_minus_bins[bin_index] += xi_epsG_m_ij
                    
                    xi_gamF_plus_autoVar_bins[bin_index] += xi_epsF_p_aV_ij
                    xi_gamF_minus_autoVar_bins[bin_index] += xi_epsF_m_aV_ij
                    xi_Fgam_plus_autoVar_bins[bin_index] += xi_Feps_p_aV_ij
                    xi_Fgam_minus_autoVar_bins[bin_index] += xi_Feps_m_aV_ij
                    xi_Ggam_plus_autoVar_bins[bin_index] += xi_Geps_p_aV_ij
                    xi_Ggam_minus_autoVar_bins[bin_index] += xi_Geps_m_aV_ij
                    xi_gamG_plus_autoVar_bins[bin_index] += xi_epsG_p_aV_ij
                    xi_gamG_minus_autoVar_bins[bin_index] += xi_epsG_m_aV_ij

                    xi_gamF_cross1_bins[bin_index] += xi_epsF_c1_ij
                    xi_gamF_cross2_bins[bin_index] += xi_epsF_c2_ij
                    xi_Fgam_cross1_bins[bin_index] += xi_Feps_c1_ij
                    xi_Fgam_cross2_bins[bin_index] += xi_Feps_c2_ij
                    xi_Ggam_cross1_bins[bin_index] += xi_Geps_c1_ij
                    xi_Ggam_cross2_bins[bin_index] += xi_Geps_c2_ij
                    xi_gamG_cross1_bins[bin_index] += xi_epsG_c1_ij
                    xi_gamG_cross2_bins[bin_index] += xi_epsG_c2_ij


        xi_gamF_plus = xi_gamF_plus_bins/Np
        xi_gamF_minus = xi_gamF_minus_bins/Np
        xi_Fgam_plus = xi_Fgam_plus_bins/Np
        xi_Fgam_minus = xi_Fgam_minus_bins/Np
        xi_Ggam_plus = xi_Ggam_plus_bins/Np
        xi_Ggam_minus = xi_Ggam_minus_bins/Np
        xi_gamG_plus = xi_gamG_plus_bins/Np
        xi_gamG_minus = xi_gamG_minus_bins/Np

        xi_gamF_plus_autoVar = xi_gamF_plus_autoVar_bins*(self.sigma_eps**2.*self.sigma_aF**2./(2*Np**2.))
        xi_gamF_minus_autoVar = xi_gamF_minus_autoVar_bins*(self.sigma_eps**2.*self.sigma_aF**2./(2*Np**2.))
        xi_Fgam_plus_autoVar = xi_Fgam_plus_autoVar_bins*(self.sigma_eps**2.*self.sigma_aF**2./(2*Np**2.))
        xi_Fgam_minus_autoVar = xi_Fgam_minus_autoVar_bins*(self.sigma_eps**2.*self.sigma_aF**2./(2*Np**2.))
        xi_Ggam_plus_autoVar = xi_Ggam_plus_autoVar_bins*(self.sigma_eps**2.*self.sigma_aG**2./(2*Np**2.))
        xi_Ggam_minus_autoVar = xi_Ggam_minus_autoVar_bins*(self.sigma_eps**2.*self.sigma_aG**2./(2*Np**2.))
        xi_gamG_plus_autoVar = xi_gamG_plus_autoVar_bins*(self.sigma_eps**2.*self.sigma_aG**2./(2*Np**2.))
        xi_gamG_minus_autoVar = xi_gamG_minus_autoVar_bins*(self.sigma_eps**2.*self.sigma_aG**2./(2*Np**2.))

        xi_gamF_cross1 = xi_gamF_cross1_bins/Np
        xi_gamF_cross2 = xi_gamF_cross2_bins/Np
        xi_Fgam_cross1 = xi_Fgam_cross1_bins/Np
        xi_Fgam_cross2 = xi_Fgam_cross2_bins/Np
        xi_Ggam_cross1 = xi_Ggam_cross1_bins/Np
        xi_Ggam_cross2 = xi_Ggam_cross2_bins/Np
        xi_gamG_cross1 = xi_gamG_cross1_bins/Np
        xi_gamG_cross2 = xi_gamG_cross2_bins/Np

        # Convert shear-flexions from 1/arcsec to 1/rad
        xi_gamF_plus = (xi_gamF_plus/u.arcsec).to(1/u.rad).value
        xi_gamF_minus = (xi_gamF_minus/u.arcsec).to(1/u.rad).value
        xi_Fgam_plus = (xi_Fgam_plus/u.arcsec).to(1/u.rad).value
        xi_Fgam_minus = (xi_Fgam_minus/u.arcsec).to(1/u.rad).value
        xi_Ggam_plus = (xi_Ggam_plus/u.arcsec).to(1/u.rad).value
        xi_Ggam_minus = (xi_Ggam_minus/u.arcsec).to(1/u.rad).value
        xi_gamG_plus = (xi_gamG_plus/u.arcsec).to(1/u.rad).value
        xi_gamG_minus = (xi_gamG_minus/u.arcsec).to(1/u.rad).value

        xi_gamF_cross1 = (xi_gamF_cross1/u.arcsec).to(1/u.rad).value
        xi_gamF_cross2 = (xi_gamF_cross2/u.arcsec).to(1/u.rad).value
        xi_Fgam_cross1 = (xi_Fgam_cross1/u.arcsec).to(1/u.rad).value
        xi_Fgam_cross2 = (xi_Fgam_cross2/u.arcsec).to(1/u.rad).value
        xi_Ggam_cross1 = (xi_Ggam_cross1/u.arcsec).to(1/u.rad).value
        xi_Ggam_cross2 = (xi_Ggam_cross2/u.arcsec).to(1/u.rad).value
        xi_gamG_cross1 = (xi_gamG_cross1/u.arcsec).to(1/u.rad).value
        xi_gamG_cross2 = (xi_gamG_cross2/u.arcsec).to(1/u.rad).value

        xi_gamF_plus_autoVar = (xi_gamF_plus_autoVar/u.arcsec**2.).to(1/u.rad**2.).value
        xi_gamF_minus_autoVar = (xi_gamF_minus_autoVar/u.arcsec**2.).to(1/u.rad**2.).value
        xi_Fgam_plus_autoVar = (xi_Fgam_plus_autoVar/u.arcsec**2.).to(1/u.rad**2.).value
        xi_Fgam_minus_autoVar = (xi_Fgam_minus_autoVar/u.arcsec**2.).to(1/u.rad**2.).value
        xi_Ggam_plus_autoVar = (xi_Ggam_plus_autoVar/u.arcsec**2.).to(1/u.rad**2.).value
        xi_Ggam_minus_autoVar = (xi_Ggam_minus_autoVar/u.arcsec**2.).to(1/u.rad**2.).value
        xi_gamG_plus_autoVar = (xi_gamG_plus_autoVar/u.arcsec**2.).to(1/u.rad**2.).value
        xi_gamG_minus_autoVar = (xi_gamG_minus_autoVar/u.arcsec**2.).to(1/u.rad**2.).value

        # .. Export shear-flexion correlation functions to .pkl file
        col_list = ['theta', 'Np', 
                    'xi_gamF_plus', 'xi_gamF_plus_autoVar', 'xi_gamF_minus', 'xi_gamF_minus_autoVar', 
                    'xi_Fgam_plus', 'xi_Fgam_plus_autoVar', 'xi_Fgam_minus', 'xi_Fgam_minus_autoVar', 
                    'xi_Ggam_plus', 'xi_Ggam_plus_autoVar', 'xi_Ggam_minus', 'xi_Ggam_minus_autoVar', 
                    'xi_gamG_plus', 'xi_gamG_plus_autoVar', 'xi_gamG_minus', 'xi_gamG_minus_autoVar'
                    'xi_gamF_cross1', 'xi_gamF_cross2',
                    'xi_Fgam_cross1', 'xi_Fgam_cross2',
                    'xi_Ggam_cross1', 'xi_Ggam_cross2',
                    'xi_gamG_cross1', 'xi_gamG_cross2']
        arrs = [theta_list, Np, 
                xi_gamF_plus, xi_gamF_plus_autoVar, xi_gamF_minus, xi_gamF_minus_autoVar, 
                xi_Fgam_plus, xi_Fgam_plus_autoVar, xi_Fgam_minus, xi_Fgam_minus_autoVar, 
                xi_Ggam_plus, xi_Ggam_plus_autoVar, xi_Ggam_minus, xi_Ggam_minus_autoVar, 
                xi_gamG_plus, xi_gamG_plus_autoVar, xi_gamG_minus, xi_gamG_minus_autoVar, 
                xi_gamF_cross1, xi_gamF_cross2,
                xi_Fgam_cross1, xi_Fgam_cross2,
                xi_Ggam_cross1, xi_Ggam_cross2,
                xi_gamG_cross1, xi_gamG_cross2]
        dat = {i:arrs[j] for i,j in zip(col_list, range(len(col_list)))}
        out_frame = pd.DataFrame(data = dat, columns = col_list)
        #out_frame.to_pickle(self.survey+'/'+self.survey+'_Measure/shear-flexion_two_point_measured_'+self.survey+'_bin_combo_'+self.bin_combo+'.pkl') 
        out_frame.to_csv('shear-flexion_two_point_measured_'+self.survey+'_bin_combo_'+self.bin_combo+'.csv')
    
    
    def theta_ij(self, xi,yi,xj,yj):
        """Get the magnitude of the angular separation vector, 
             varTheta_ij = |varTheta_j - varTheta_i|,
           between two galaxies i and j (in arcsec)
        """
        varTheta = np.sqrt((xj-xi)**2.+(yj-yi)**2.)
        return varTheta

    def varPhi_ij(self, xi,yi,xj,yj):
        """Get the polar angle of the angular separation vector,
           between two galaxies i and j (in radians)
        """
        varPhi = np.arctan2((yj-yi), (xj-xi))
        return varPhi

    def F1_rot(self, F1, F2, varPhi):
        """Get the rotated 1-component of the F-flexion, defined as
             F1_rot = -Re(Fe^(-i*varPhi)),
           where F = F1 + iF2, and phi is the polar angle of the separation vector theta
        """
        F1_r = -F1*np.cos(varPhi) - F2*np.sin(varPhi)
        return F1_r

    def F2_rot(self, F1, F2, varPhi):
        """Get the rotated 2-component of the F-flexion, defined as
             F2_rot = -Im(Fe^(-i*varPhi)),
           where F = F1 + iF2, and phi is the polar angle of the separation vector theta
        """
        F2_r = -F2*np.cos(varPhi) + F1*np.sin(varPhi)
        return F2_r

    def G1_rot(self, G1, G2, varPhi):
        """Get the rotated 1-component of the G-flexion, defined as
             G1_rot = -Re(Fe^(-i*3*varPhi)),
           where G = G1 + iG2, and phi is the polar angle of the separation vector theta
        """
        G1_r = G1*np.cos(3*varPhi) + G2*np.sin(3*varPhi)
        return G1_r

    def G2_rot(self, G1, G2, varPhi):
        """Get the rotated 2-component of the G-flexion, defined as
             G2_rot = -Im(Fe^(-i*3*varPhi)),
           where G = G1 + iG2, and phi is the polar angle of the separation vector theta
        """
        G2_r = G2*np.cos(3*varPhi) - G1*np.sin(3*varPhi)
        return G2_r

    def eps1_rot(self, eps1, eps2, varPhi):
        """Get the rotated 1-component of the ellipticity, defined as
             eps1_rot = -Re(eps*e^(-2i*varPhi)),
           where eps = eps1 + ieps2, and phi is the polar angle of the separation vector theta
        """
        eps1_r = -eps1*np.cos(2*varPhi) - eps2*np.sin(2*varPhi)
        return eps1_r

    def eps2_rot(self, eps1, eps2, varPhi):
        """Get the rotated 2-component of the ellipticity, defined as
             eps_x = -Im(eps*e^(-2i*varPhi)),
           where eps = eps1 + ieps2, and phi is the polar angle of the separation vector theta
        """
        eps2_r = -eps2*np.cos(2*varPhi) + eps1*np.sin(2*varPhi)
        return eps2_r

    def get_intrinsic_scatter(self):
        """Get the intrinsic flexion and intrinsic ellipticity, to be used in the
           covariance/autovariance calculation.
        """
        # Intrinsic first flexion
        F = np.sqrt(self.F1_list**2. + self.F2_list**2.)
        sigma_aF = np.sqrt(np.mean((self.a_list*F*self.w_list)**2.))
        # Intrinsic second flexion
        G = np.sqrt(self.G1_list**2. + self.G2_list**2.)
        sigma_aG = np.sqrt(np.mean((self.a_list*G*self.w_list)**2.))
        # Intrinsic ellipticity
        eps = np.sqrt(self.eps1_list**2. + self.eps2_list**2.)
        sigma_eps = np.sqrt(np.mean((eps*self.w_list)**2.))

        return sigma_aF, sigma_aG, sigma_eps


        


















