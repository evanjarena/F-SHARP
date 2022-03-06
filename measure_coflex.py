import numpy as np
import sys,os
from astropy.cosmology import Planck15
import pandas as pd
from astropy import units as u
from scipy import interpolate

from fastdist import fastdist



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
        print('sigma_aF, sigma_aG, sigma_eps =', sigma_aF, sigma_aG, sigma_eps)
        self.sigma_aF = sigma_aF
        self.sigma_aG = sigma_aG
        self.sigma_eps = sigma_eps

        # Get flexion-flexion correlation functions first. 
        # .. Get angular separations and bins
        theta_flexflex_list, flexflex_bins = self.theta_flexflex_bin()
        # .. Get weights and two-point arrays in each bin
        # .. .. flexflex_bins = array(wp_bins, xi_FF_plus_bins, ...)
        flexflex_bins = self.get_binned_two_point_flexflex(theta_flexflex_list, flexflex_bins)
        # .. Get two-point correlation functions and autovariances
        flexflex = self.get_flexflex_corr(*flexflex_bins)
        xi_FF_plus = flexflex[0] 
        xi_FF_plus_autoVar = flexflex[1] 
        xi_FF_minus = flexflex[2] 
        xi_FF_minus_autoVar = flexflex[3]
        xi_GG_plus = flexflex[4] 
        xi_GG_plus_autoVar = flexflex[5] 
        xi_GG_minus = flexflex[6] 
        xi_GG_minus_autoVar = flexflex[7] 
        xi_FG_plus = flexflex[8] 
        xi_FG_plus_autoVar = flexflex[9] 
        xi_FG_minus = flexflex[10] 
        xi_FG_minus_autoVar = flexflex[11]
        xi_GF_plus = flexflex[12] 
        xi_GF_plus_autoVar = flexflex[13] 
        xi_GF_minus = flexflex[14] 
        xi_GF_minus_autoVar = flexflex[15]  
        # .. Export flexion-flexion correlation functions to .pkl file
        col_list = ['theta', 'xi_FF_plus', 'xi_FF_plus_autoVar', 'xi_FF_minus', 'xi_FF_minus_autoVar', 'xi_GG_plus', 'xi_GG_plus_autoVar', 'xi_GG_minus', 'xi_GG_minus_autoVar', 'xi_FG_plus', 'xi_FG_plus_autoVar', 'xi_FG_minus', 'xi_FG_minus_autoVar', 'xi_GF_plus', 'xi_GF_plus_autoVar', 'xi_GF_minus', 'xi_GF_minus_autoVar']
        arrs = [theta_flexflex_list, xi_FF_plus, xi_FF_plus_autoVar, xi_FF_minus, xi_FF_minus_autoVar, xi_GG_plus, xi_GG_plus_autoVar, xi_GG_minus, xi_GG_minus_autoVar, xi_FG_plus, xi_FG_plus_autoVar, xi_FG_minus, xi_FG_minus_autoVar, xi_GF_plus, xi_GF_plus_autoVar, xi_GF_minus, xi_GF_minus_autoVar]
        dat = {i:arrs[j] for i,j in zip(col_list, range(len(col_list)))}
        out_frame = pd.DataFrame(data = dat, columns = col_list)
        out_frame.to_pickle(self.survey+'/'+self.survey+'_Measure/flexion-flexion_two_point_measured_'+self.survey+'_bin_combo_'+self.bin_combo+'.pkl')

        # Shear-flexion correlations
        # .. Get angular separations and bins
        theta_shearflex_list, shearflex_bins = self.theta_shearflex_bin()
        # .. Get weights and two-point arrays in each bin
        # .. .. flexflex_bins = array(wp_bins, xi_FF_plus_bins, ...)
        shearflex_bins = self.get_binned_two_point_shearflex(theta_shearflex_list, shearflex_bins)
        # .. Get two-point correlation functions
        shearflex = self.get_shearflex_corr(*shearflex_bins)
        xi_gamF_plus = shearflex[0]
        xi_gamF_plus_autoVar = shearflex[1]
        xi_gamF_minus = shearflex[2]
        xi_gamF_minus_autoVar = shearflex[3]
        xi_Fgam_plus = shearflex[4]
        xi_Fgam_plus_autoVar = shearflex[5]
        xi_Fgam_minus = shearflex[6]
        xi_Fgam_minus_autoVar = shearflex[7]
        xi_Ggam_plus = shearflex[8]
        xi_Ggam_plus_autoVar = shearflex[9]
        xi_Ggam_minus = shearflex[10]
        xi_Ggam_minus_autoVar = shearflex[11]
        xi_gamG_plus = shearflex[12]
        xi_gamG_plus_autoVar = shearflex[13]
        xi_gamG_minus = shearflex[14]
        xi_gamG_minus_autoVar = shearflex[15]
        # .. Export shear-flexion correlation functions to .pkl file
        col_list = ['theta', 'xi_gamF_plus', 'xi_gamF_plus_autoVar', 'xi_gamF_minus', 'xi_gamF_minus_autoVar', 'xi_Fgam_plus', 'xi_Fgam_plus_autoVar', 'xi_Fgam_minus', 'xi_Fgam_minus_autoVar', 'xi_Ggam_plus', 'xi_Ggam_plus_autoVar', 'xi_Ggam_minus', 'xi_Ggam_minus_autoVar', 'xi_gamG_plus', 'xi_gamG_plus_autoVar', 'xi_gamG_minus', 'xi_gamG_minus_autoVar']
        arrs = [theta_shearflex_list, xi_gamF_plus, xi_gamF_plus_autoVar, xi_gamF_minus, xi_gamF_minus_autoVar,  xi_Fgam_plus, xi_Fgam_plus_autoVar, xi_Fgam_minus, xi_Fgam_minus_autoVar, xi_Ggam_plus, xi_Ggam_plus_autoVar, xi_Ggam_minus, xi_Ggam_minus_autoVar, xi_gamG_plus, xi_gamG_plus_autoVar, xi_gamG_minus, xi_gamG_minus_autoVar]
        dat = {i:arrs[j] for i,j in zip(col_list, range(len(col_list)))}
        out_frame = pd.DataFrame(data = dat, columns = col_list)
        out_frame.to_pickle(self.survey+'/'+self.survey+'_Measure/shear-flexion_two_point_measured_'+self.survey+'_bin_combo_'+self.bin_combo+'.pkl') 

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
        # .. Product of weights
        wp_bins = [[] for _ in range(N_bins)]
        # .. Auto-correlations
        xi_FF_plus_bins = [[] for _ in range(N_bins)]
        xi_FF_minus_bins = [[] for _ in range(N_bins)]
        xi_FF_plus_autoVar_bins = [[] for _ in range(N_bins)]
        xi_FF_minus_autoVar_bins = [[] for _ in range(N_bins)]
        xi_GG_plus_bins = [[] for _ in range(N_bins)]
        xi_GG_minus_bins = [[] for _ in range(N_bins)]
        xi_GG_plus_autoVar_bins = [[] for _ in range(N_bins)]
        xi_GG_minus_autoVar_bins = [[] for _ in range(N_bins)]
        # .. Cross-correlations
        xi_FG_plus_bins = [[] for _ in range(N_bins)]
        xi_FG_minus_bins = [[] for _ in range(N_bins)]
        xi_FG_plus_autoVar_bins = [[] for _ in range(N_bins)]
        xi_FG_minus_autoVar_bins = [[] for _ in range(N_bins)]
        xi_GF_plus_bins = [[] for _ in range(N_bins)]
        xi_GF_minus_bins = [[] for _ in range(N_bins)]
        xi_GF_plus_autoVar_bins = [[] for _ in range(N_bins)]
        xi_GF_minus_autoVar_bins = [[] for _ in range(N_bins)]

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

            # Get grid coordinates for galaxy i
            gx_i = gx_list[i]
            gy_i = gy_list[i]
            # Get (RA,Dec) for galaxy i
            x_i = self.x_list[i]
            y_i = self.y_list[i]
            # Get size for galaxy i
            a_i = self.a_list[i]
            a_i *= (u.arcsec)
            a_i = a_i.to(u.rad).value  
            # Get weight of galaxy i
            w_i = self.w_list[i]
            # Get flexion of galaxy i
            F1_i = self.F1_list[i]
            F2_i = self.F2_list[i]
            G1_i = self.G1_list[i]
            G2_i = self.G2_list[i]

            # Get galaxy pairs {j} associated with galaxy i
            id = np.where((j_list > i) &
                          (gx_list <= gx_i+1) & (gx_list >= gx_i-1) &
                          (gy_list <= gy_i+1) & (gy_list >= gy_i-1))
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
                theta_ij = self.theta_ij(x_i,y_i,x_j_list[j],y_j_list[j])
                # Get polar angle between (i,j)
                varPhi_ij = self.varPhi_ij(x_i,y_i,x_j_list[j],y_j_list[j])
                # Calculate tangential and radial flexions for pair (i,j)
                F1_rot_i = self.F1_rot(F1_i, F2_i, varPhi_ij)
                F1_rot_j = self.F1_rot(F1_j_list[j], F2_j_list[j], varPhi_ij)
                F2_rot_i = self.F2_rot(F1_i, F2_i, varPhi_ij)
                F2_rot_j = self.F2_rot(F1_j_list[j], F2_j_list[j], varPhi_ij)
                G1_rot_i = self.G1_rot(G1_i, G2_i, varPhi_ij)
                G1_rot_j = self.G1_rot(G1_j_list[j], G2_j_list[j], varPhi_ij)
                G2_rot_i = self.G2_rot(G1_i, G2_i, varPhi_ij)
                G2_rot_j = self.G2_rot(G1_j_list[j], G2_j_list[j], varPhi_ij)
                # Convert the flexions from 1/arcsec to 1/rad:
                F1_rot_i /= (u.arcsec)
                F1_rot_i = F1_rot_i.to(1/u.rad).value
                F1_rot_j /= (u.arcsec)
                F1_rot_j = F1_rot_j.to(1/u.rad).value
                F2_rot_i /= (u.arcsec)
                F2_rot_i = F2_rot_i.to(1/u.rad).value
                F2_rot_j /= (u.arcsec)
                F2_rot_j = F2_rot_j.to(1/u.rad).value
                G1_rot_i /= (u.arcsec)
                G1_rot_i = G1_rot_i.to(1/u.rad).value
                G1_rot_j /= (u.arcsec)
                G1_rot_j = G1_rot_j.to(1/u.rad).value
                G2_rot_i /= (u.arcsec)
                G2_rot_i = G2_rot_i.to(1/u.rad).value
                G2_rot_j /= (u.arcsec)
                G2_rot_j = G2_rot_j.to(1/u.rad).value
                # Convert size to rad
                a_j = a_j_list[j]*(u.arcsec)
                a_j = a_j.to(u.rad).value
                # Weight for each pair
                wp_ij = w_i*w_j_list[j]
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
                xi_FF_p_aV_ij = (wp_ij/(a_i*a_j))**2.
                xi_FF_m_aV_ij = xi_FF_p_aV_ij
                xi_GG_p_aV_ij = xi_FF_p_aV_ij
                xi_GG_m_aV_ij = xi_FF_p_aV_ij
                xi_FG_p_aV_ij = xi_FF_p_aV_ij
                xi_FG_m_aV_ij = xi_FF_p_aV_ij
                xi_GF_p_aV_ij = xi_FF_p_aV_ij
                xi_GF_m_aV_ij = xi_FF_p_aV_ij

                # Get the bin for each galaxy pair (i,j).  It is simplest to use
                # np.digitize(theta_ij, bins).  This returns the bin number that 
                # theta_ij belongs to (digitize indexes at 1). If digitize returns 0,
                # theta_ij is smaller than the smallest bin.  If digitize returns the 
                # number = N_bins + 1, then theta_ij is larger than the largest bin. So
                bin_ij = np.digitize(theta_ij, bins, right=True)
                if (bin_ij > 0) & (bin_ij < N_bins + 1):
                    bin_index = bin_ij-1
                    wp_bins[bin_index].append(wp_ij)
                    xi_FF_plus_bins[bin_index].append(xi_FF_p_ij)
                    xi_FF_minus_bins[bin_index].append(xi_FF_m_ij)
                    xi_GG_plus_bins[bin_index].append(xi_GG_p_ij)
                    xi_GG_minus_bins[bin_index].append(xi_GG_m_ij)
                    xi_FG_plus_bins[bin_index].append(xi_FG_p_ij)
                    xi_FG_minus_bins[bin_index].append(xi_FG_m_ij)
                    xi_GF_plus_bins[bin_index].append(xi_GF_p_ij)
                    xi_GF_minus_bins[bin_index].append(xi_GF_m_ij)

                    xi_FF_plus_autoVar_bins[bin_index].append(xi_FF_p_aV_ij)
                    xi_FF_minus_autoVar_bins[bin_index].append(xi_FF_m_aV_ij)
                    xi_GG_plus_autoVar_bins[bin_index].append(xi_GG_p_aV_ij)
                    xi_GG_minus_autoVar_bins[bin_index].append(xi_GG_m_aV_ij)
                    xi_FG_plus_autoVar_bins[bin_index].append(xi_FG_p_aV_ij)
                    xi_FG_minus_autoVar_bins[bin_index].append(xi_FG_m_aV_ij)
                    xi_GF_plus_autoVar_bins[bin_index].append(xi_GF_p_aV_ij)
                    xi_GF_minus_autoVar_bins[bin_index].append(xi_GF_m_aV_ij)
        
        return wp_bins, xi_FF_plus_bins, xi_FF_minus_bins, xi_GG_plus_bins, xi_GG_minus_bins,  xi_FG_plus_bins, xi_FG_minus_bins,  xi_GF_plus_bins, xi_GF_minus_bins, xi_FF_plus_autoVar_bins, xi_FF_minus_autoVar_bins, xi_GG_plus_autoVar_bins, xi_GG_minus_autoVar_bins, xi_FG_plus_autoVar_bins, xi_FG_minus_autoVar_bins, xi_GF_plus_autoVar_bins, xi_GF_minus_autoVar_bins,

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
        wp_bins = [[] for _ in range(N_bins)]
        xi_gamF_plus_bins = [[] for _ in range(N_bins)]
        xi_gamF_minus_bins = [[] for _ in range(N_bins)]
        xi_Fgam_plus_bins = [[] for _ in range(N_bins)]
        xi_Fgam_minus_bins = [[] for _ in range(N_bins)]
        xi_Ggam_plus_bins = [[] for _ in range(N_bins)]
        xi_Ggam_minus_bins = [[] for _ in range(N_bins)]
        xi_gamG_plus_bins = [[] for _ in range(N_bins)]
        xi_gamG_minus_bins = [[] for _ in range(N_bins)]
        
        xi_gamF_plus_autoVar_bins = [[] for _ in range(N_bins)]
        xi_gamF_minus_autoVar_bins = [[] for _ in range(N_bins)]
        xi_Fgam_plus_autoVar_bins = [[] for _ in range(N_bins)]
        xi_Fgam_minus_autoVar_bins = [[] for _ in range(N_bins)]
        xi_Ggam_plus_autoVar_bins = [[] for _ in range(N_bins)]
        xi_Ggam_minus_autoVar_bins = [[] for _ in range(N_bins)]
        xi_gamG_plus_autoVar_bins = [[] for _ in range(N_bins)]
        xi_gamG_minus_autoVar_bins = [[] for _ in range(N_bins)]
        
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

            # Get grid coordinates for galaxy i
            gx_i = gx_list[i]
            gy_i = gy_list[i]
            # Get (RA,Dec) for galaxy i
            x_i = x_list[i]
            y_i = y_list[i]
            # Get size for galaxy i
            a_i = self.a_list[i]
            a_i *= (u.arcsec)
            a_i = a_i.to(u.rad).value 
            # Get weight of galaxy i
            w_i = self.w_list[i]
            # Get flexion of galaxy i
            F1_i = self.F1_list[i]
            F2_i = self.F2_list[i]
            G1_i = self.G1_list[i]
            G2_i = self.G2_list[i]
            # Get ellipticity of galaxy i
            eps1_i = self.eps1_list[i]
            eps2_i = self.eps2_list[i]

            # Get galaxy pairs {j} associated with galaxy i
            id = np.where((j_list > i) &
                          (gx_list <= gx_i+1) & (gx_list >= gx_i-1) &
                          (gy_list <= gy_i+1) & (gy_list >= gy_i-1))
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
                theta_ij = self.theta_ij(x_i,y_i,x_j_list[j],y_j_list[j])
                # Get polar angle between (i,j)
                varPhi_ij = self.varPhi_ij(x_i,y_i,x_j_list[j],y_j_list[j])
                # Calculate tangential and radial flexions for pair (i,j)
                F1_rot_i = self.F1_rot(F1_i, F2_i, varPhi_ij)
                F1_rot_j = self.F1_rot(F1_j_list[j], F2_j_list[j], varPhi_ij)
                F2_rot_i = self.F2_rot(F1_i, F2_i, varPhi_ij)
                F2_rot_j = self.F2_rot(F1_j_list[j], F2_j_list[j], varPhi_ij)
                G1_rot_i = self.G1_rot(G1_i, G2_i, varPhi_ij)
                G1_rot_j = self.G1_rot(G1_j_list[j], G2_j_list[j], varPhi_ij)
                G2_rot_i = self.G2_rot(G1_i, G2_i, varPhi_ij)
                G2_rot_j = self.G2_rot(G1_j_list[j], G2_j_list[j], varPhi_ij)
                # Calculate tangential and cross ellipticities for pair (i,j)
                eps1_rot_i = self.eps1_rot(eps1_i, eps2_i, varPhi_ij)
                eps1_rot_j = self.eps1_rot(eps1_j_list[j], eps2_j_list[j], varPhi_ij)
                eps2_rot_i = self.eps2_rot(eps1_i, eps2_i, varPhi_ij)
                eps2_rot_j = self.eps2_rot(eps1_j_list[j], eps2_j_list[j], varPhi_ij)
                # Convert the flexions from 1/arcsec to 1/rad:
                F1_rot_i /= (u.arcsec)
                F1_rot_i = F1_rot_i.to(1/u.rad).value
                F1_rot_j /= (u.arcsec)
                F1_rot_j = F1_rot_j.to(1/u.rad).value
                F2_rot_i /= (u.arcsec)
                F2_rot_i = F2_rot_i.to(1/u.rad).value
                F2_rot_j /= (u.arcsec)
                F2_rot_j = F2_rot_j.to(1/u.rad).value
                G1_rot_i /= (u.arcsec)
                G1_rot_i = G1_rot_i.to(1/u.rad).value
                G1_rot_j /= (u.arcsec)
                G1_rot_j = G1_rot_j.to(1/u.rad).value
                G2_rot_i /= (u.arcsec)
                G2_rot_i = G2_rot_i.to(1/u.rad).value
                G2_rot_j /= (u.arcsec)
                G2_rot_j = G2_rot_j.to(1/u.rad).value
                # Convert size to rad
                a_j = a_j_list[j]*(u.arcsec)
                a_j = a_j.to(u.rad).value
                # Weight for each pair
                wp_ij = w_i*w_j_list[j]
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
                xi_epsF_p_aV_ij = (wp_ij/(a_j))**2.
                xi_epsF_m_aV_ij = (wp_ij/(a_j))**2.
                xi_Feps_p_aV_ij = (wp_ij/(a_i))**2.
                xi_Feps_m_aV_ij = (wp_ij/(a_i))**2.
                xi_Geps_p_aV_ij = (wp_ij/(a_i))**2.
                xi_Geps_m_aV_ij = (wp_ij/(a_i))**2.
                xi_epsG_p_aV_ij = (wp_ij/(a_j))**2.
                xi_epsG_m_aV_ij = (wp_ij/(a_j))**2.

                # Get the bin for each galaxy pair (i,j).  It is simplest to use
                # np.digitize(theta_ij, bins).  This returns the bin number that 
                # theta_ij belongs to (digitize indexes at 1). If digitize returns 0,
                # theta_ij is smaller than the smallest bin.  If digitize returns the 
                # number = N_bins + 1, then theta_ij is larger than the largest bin. So
                bin_ij = np.digitize(theta_ij, bins, right=True)
                if (bin_ij > 0) & (bin_ij < N_bins + 1):
                    bin_index = bin_ij-1
                    wp_bins[bin_index].append(wp_ij)
                    xi_gamF_plus_bins[bin_index].append(xi_epsF_p_ij)
                    xi_gamF_minus_bins[bin_index].append(xi_epsF_m_ij)
                    xi_Fgam_plus_bins[bin_index].append(xi_Feps_p_ij)
                    xi_Fgam_minus_bins[bin_index].append(xi_Feps_m_ij)
                    xi_Ggam_plus_bins[bin_index].append(xi_Geps_p_ij)
                    xi_Ggam_minus_bins[bin_index].append(xi_Geps_m_ij)
                    xi_gamG_plus_bins[bin_index].append(xi_epsG_p_ij)
                    xi_gamG_minus_bins[bin_index].append(xi_epsG_m_ij)
                    
                    xi_gamF_plus_autoVar_bins[bin_index].append(xi_epsF_p_aV_ij)
                    xi_gamF_minus_autoVar_bins[bin_index].append(xi_epsF_m_aV_ij)
                    xi_Fgam_plus_autoVar_bins[bin_index].append(xi_Feps_p_aV_ij)
                    xi_Fgam_minus_autoVar_bins[bin_index].append(xi_Feps_m_aV_ij)
                    xi_Ggam_plus_autoVar_bins[bin_index].append(xi_Geps_p_aV_ij)
                    xi_Ggam_minus_autoVar_bins[bin_index].append(xi_Geps_m_aV_ij)
                    xi_gamG_plus_autoVar_bins[bin_index].append(xi_epsG_p_aV_ij)
                    xi_gamG_minus_autoVar_bins[bin_index].append(xi_epsG_m_aV_ij)
        
        return wp_bins, xi_gamF_plus_bins, xi_gamF_minus_bins, xi_Fgam_plus_bins, xi_Fgam_minus_bins, xi_Ggam_plus_bins, xi_Ggam_minus_bins, xi_gamG_plus_bins, xi_gamG_minus_bins, xi_gamF_plus_autoVar_bins, xi_gamF_minus_autoVar_bins, xi_Fgam_plus_autoVar_bins, xi_Fgam_minus_autoVar_bins, xi_Ggam_plus_autoVar_bins, xi_Ggam_minus_autoVar_bins, xi_gamG_plus_autoVar_bins, xi_gamG_minus_autoVar_bins 

    # To Do: finish below

    def get_flexflex_corr(self, wp_bins, 
                            xi_FF_plus_bins, xi_FF_minus_bins, 
                            xi_GG_plus_bins, xi_GG_minus_bins,
                            xi_FG_plus_bins, xi_FG_minus_bins, 
                            xi_GF_plus_bins, xi_GF_minus_bins, 
                            xi_FF_plus_autoVar_bins, xi_FF_minus_autoVar_bins, 
                            xi_GG_plus_autoVar_bins, xi_GG_minus_autoVar_bins,
                            xi_FG_plus_autoVar_bins, xi_FG_minus_autoVar_bins, 
                            xi_GF_plus_autoVar_bins, xi_GF_minus_autoVar_bins):
        """
        Calculate the two-point correlation functions and their errors
        within each bin.
        """
        # Get number of bins
        N_bins = len(wp_bins)
        # Initiliaze arrays for two-point functions and errors
        xi_FF_plus_list = []
        xi_FF_minus_list = []
        xi_FF_plus_autoVar_list = []
        xi_FF_minus_autoVar_list = []
        xi_GG_plus_list = []
        xi_GG_minus_list = []
        xi_GG_plus_autoVar_list = []
        xi_GG_minus_autoVar_list = []
        xi_FG_plus_list = []
        xi_FG_minus_list = []
        xi_FG_plus_autoVar_list = []
        xi_FG_minus_autoVar_list = []
        xi_GF_plus_list = []
        xi_GF_minus_list = []
        xi_GF_plus_autoVar_list = []
        xi_GF_minus_autoVar_list = []

        for i in range(N_bins):
            # Get number of pairs in bin i
            Np = np.sum(wp_bins[i])
            # Get two point correlation functions and autovariances
            # .. FF
            print('Np =', Np)
            xi_FF_p = np.sum(xi_FF_plus_bins[i])/Np
            xi_FF_p_aV = (self.sigma_aF**4./(2*Np**2.))*np.sum(xi_FF_plus_autoVar_bins[i])
            xi_FF_m = np.sum(xi_FF_minus_bins[i])/Np
            xi_FF_m_aV = (self.sigma_aF**4./(2*Np**2.))*np.sum(xi_FF_minus_autoVar_bins[i])
            xi_FF_plus_list.append(xi_FF_p)
            xi_FF_minus_list.append(xi_FF_m)
            xi_FF_plus_autoVar_list.append(xi_FF_p_aV)
            xi_FF_minus_autoVar_list.append(xi_FF_m_aV)
            # .. GG
            xi_GG_p = np.sum(xi_GG_plus_bins[i])/Np
            xi_GG_p_aV = (self.sigma_aG**4./(2*Np**2.))*np.sum(xi_GG_plus_autoVar_bins[i])
            xi_GG_m = np.sum(xi_GG_minus_bins[i])/Np
            xi_GG_m_aV = (self.sigma_aG**4./(2*Np**2.))*np.sum(xi_GG_minus_autoVar_bins[i])
            xi_GG_plus_list.append(xi_GG_p)
            xi_GG_minus_list.append(xi_GG_m)
            xi_GG_plus_autoVar_list.append(xi_GG_p_aV)
            xi_GG_minus_autoVar_list.append(xi_GG_m_aV)
            # .. FG
            xi_FG_p = np.sum(xi_FG_plus_bins[i])/Np
            xi_FG_p_aV = (self.sigma_aF**2.*self.sigma_aG**2./(2*Np**2.))*np.sum(xi_FG_plus_autoVar_bins[i])
            xi_FG_m = np.sum(xi_FG_minus_bins[i])/Np
            xi_FG_m_aV = (self.sigma_aF**2.*self.sigma_aG**2./(2*Np**2.))*np.sum(xi_FG_minus_autoVar_bins[i])
            xi_FG_plus_list.append(xi_FG_p)
            xi_FG_minus_list.append(xi_FG_m)
            xi_FG_plus_autoVar_list.append(xi_FG_p_aV)
            xi_FG_minus_autoVar_list.append(xi_FG_m_aV)
            # .. GF
            xi_GF_p = np.sum(xi_GF_plus_bins[i])/Np
            xi_GF_p_aV = (self.sigma_aF**2.*self.sigma_aG**2./(2*Np**2.))*np.sum(xi_GF_plus_autoVar_bins[i])
            xi_GF_m = np.sum(xi_GF_minus_bins[i])/Np
            xi_GF_m_aV = (self.sigma_aF**2.*self.sigma_aG**2./(2*Np**2.))*np.sum(xi_GF_minus_autoVar_bins[i])
            xi_GF_plus_list.append(xi_GF_p)
            xi_GF_minus_list.append(xi_GF_m)
            xi_GF_plus_autoVar_list.append(xi_GF_p_aV)
            xi_GF_minus_autoVar_list.append(xi_GF_m_aV)

        return xi_FF_plus_list, xi_FF_plus_autoVar_list, xi_FF_minus_list, xi_FF_minus_autoVar_list, xi_GG_plus_list, xi_GG_plus_autoVar_list, xi_GG_minus_list, xi_GG_minus_autoVar_list, xi_GF_plus_list, xi_GF_plus_autoVar_list, xi_GF_minus_list, xi_GF_minus_autoVar_list

    def get_shearflex_corr(self, wp_bins,
                             xi_gamF_plus_bins, xi_gamF_minus_bins, 
                             xi_gamG_plus_bins, xi_gamG_minus_bins, 
                             xi_gamF_plus_autoVar_bins, xi_gamF_minus_autoVar_bins, 
                             xi_gamG_plus_autoVar_bins, xi_gamG_minus_autoVar_bins):
        """
        Calculate the two-point correlation functions and their errors
        within each bin.
        """
        # Get number of bins
        N_bins = len(wp_bins)
        # Initiliaze arrays for two-point functions and errors
        xi_gamF_plus_list = []
        xi_gamF_minus_list = []
        xi_gamF_plus_autoVar_list = []
        xi_gamF_minus_autoVar_list = []
        xi_gamG_plus_list = []
        xi_gamG_minus_list = []
        xi_gamG_plus_autoVar_list = []
        xi_gamG_minus_autoVar_list = []

        for i in range(N_bins):
            # Get number of pairs in bin i
            Np = np.sum(wp_bins[i])
            # Get two point correlation functions and autovariances
            # .. gamF
            xi_gamF_p = np.sum(xi_gamF_plus_bins[i])/Np
            xi_gamF_p_aV = (self.sigma_eps**2.*self.sigma_aF**2./(2*Np**2.))*np.sum(xi_gamF_plus_autoVar_bins[i])
            xi_gamF_m = np.sum(xi_gamF_minus_bins[i])/Np
            xi_gamF_m_aV = (self.sigma_eps**2.*self.sigma_aF**2./(2*Np**2.))*np.sum(xi_gamF_minus_autoVar_bins[i])
            xi_gamF_plus_list.append(xi_gamF_p)
            xi_gamF_minus_list.append(xi_gamF_m)
            xi_gamF_plus_autoVar_list.append(xi_gamF_p_aV)
            xi_gamF_minus_autoVar_list.append(xi_gamF_m_aV)
            # .. Fgam
            xi_Fgam_p = np.sum(xi_Fgam_plus_bins[i])/Np
            xi_Fgam_p_aV = (self.sigma_eps**2.*self.sigma_aF**2./(2*Np**2.))*np.sum(xi_Fgam_plus_autoVar_bins[i])
            xi_Fgam_m = np.sum(xi_Fgam_minus_bins[i])/Np
            xi_Fgam_m_aV = (self.sigma_eps**2.*self.sigma_aF**2./(2*Np**2.))*np.sum(xi_Fgam_minus_autoVar_bins[i])
            xi_Fgam_plus_list.append(xi_Fgam_p)
            xi_Fgam_minus_list.append(xi_Fgam_m)
            xi_Fgam_plus_autoVar_list.append(xi_Fgam_p_aV)
            xi_Fgam_minus_autoVar_list.append(xi_Fgam_m_aV)
            # .. Ggam
            xi_Ggam_p = np.sum(xi_Ggam_plus_bins[i])/Np
            xi_Ggam_p_aV = (self.sigma_eps**2.*self.sigma_aG**2./(2*Np**2.))*np.sum(xi_Ggam_plus_autoVar_bins[i])
            xi_Ggam_m = np.sum(xi_Ggam_minus_bins[i])/Np
            xi_Ggam_m_aV = (self.sigma_eps**2.*self.sigma_aG**2./(2*Np**2.))*np.sum(xi_Ggam_minus_autoVar_bins[i])
            xi_Ggam_plus_list.append(xi_Ggam_p)
            xi_Ggam_minus_list.append(xi_Ggam_m)
            xi_Ggam_plus_autoVar_list.append(xi_Ggam_p_aV)
            xi_Ggam_minus_autoVar_list.append(xi_Ggam_m_aV)
            # .. gamG
            xi_gamG_p = np.sum(xi_gamG_plus_bins[i])/Np
            xi_gamG_p_aV = (self.sigma_eps**2.*self.sigma_aG**2./(2*Np**2.))*np.sum(xi_gamG_plus_autoVar_bins[i])
            xi_gamG_m = np.sum(xi_gamG_minus_bins[i])/Np
            xi_gamG_m_aV = (self.sigma_eps**2.*self.sigma_aG**2./(2*Np**2.))*np.sum(xi_gamG_minus_autoVar_bins[i])
            xi_gamG_plus_list.append(xi_gamG_p)
            xi_gamG_minus_list.append(xi_gamG_m)
            xi_gamG_plus_autoVar_list.append(xi_gamG_p_aV)
            xi_gamG_minus_autoVar_list.append(xi_gamG_m_aV)

        return xi_gamF_plus_list, xi_gamF_plus_autoVar_list, xi_gamF_minus_list, xi_gamF_minus_autoVar_list, xi_Fgam_plus_list, xi_Fgam_plus_autoVar_list, xi_Fgam_minus_list, xi_Fgam_minus_autoVar_list, xi_Ggam_plus_list, xi_Ggam_plus_autoVar_list, xi_Ggam_minus_list,  xi_Ggam_minus_autoVar_list, xi_gamG_plus_autoVar_list, xi_gamG_minus_list,  xi_gamG_minus_autoVar_list

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
        G1_r = -G1*np.cos(3*varPhi) - G2*np.sin(3*varPhi)
        return G1_r

    def G2_rot(self, G1, G2, varPhi):
        """Get the rotated 2-component of the G-flexion, defined as
             G2_rot = -Im(Fe^(-i*3*varPhi)),
           where G = G1 + iG2, and phi is the polar angle of the separation vector theta
        """
        G2_r = -G2*np.cos(3*varPhi) + G1*np.sin(3*varPhi)
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


        


















