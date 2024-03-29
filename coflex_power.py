import numpy as np
import pandas as pd
#from classy import Class
import pickle
import sys,os
import astropy
from astropy.cosmology import FlatLambdaCDM

import matplotlib.pyplot as plt
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)

from scipy import interpolate
from scipy import integrate

from classy import Class


class CoflexPower:

    def __init__(self, z_list, nz_bin_i, nz_bin_j, survey, bin_combo):
        self.z_list = z_list
        self.Nz = len(z_list)
        self.nz_bin_i = nz_bin_i
        self.nz_bin_j = nz_bin_j

        self.survey = str(survey)
        self.bin_combo = str(bin_combo)

        self.kmin = 1e-4
        self.kmax_class = 1e3#5e2 #1e2
        self.kmax = 1e9
        
        self.lmax = 1e11#1e6
        #self.l_list = np.array(range(2, int(self.lmax+1)))
        self.l_list = np.logspace(np.log10(1),np.log10(self.lmax),100)

    def getPower(self):
        """
        Function to be called when running matter_power in order to generate
        the nonlinear matter power spectrum.  Exports to a pickle file an array of the form
           k  P_NL(k, z_1)  P_NL(k, z_2)  ...  P_NL(k, z_n)
           .       .              .                 .
           .       .              .                 .
           .       .              .                 .
        """

        sigma8 = 0.8111
        sigma8_error = 10*0.006 #10 sigma Planck18

        # Power spectra will be calculated three times -- we want the power spectra for 
        #  the Planck 18 sigma8 value, as well as sigma8 pm 1sigma error.
        # .. sigma8
        #  First, set the cosmology
        self.cosmology()
        # Next, generate the nonlinear matter power spectrum using CLASS out to kmax_class
        self.P_NL_class()
        # Extrapolate the nonlinear matter power spectrum out to kmax
        #k_P_NL = self.P_NL_extrapolated()
        k_P_NL = self.P_NL_asymptotic()

        # Export k_P_NL(k,z) array to a pickle file
        #with open('k_P_NL.pkl', 'wb') as f:
        #    pickle.dump(k_P_NL, f)
        
        # Next, interpolate all arrays so that they can be turned into callable functions
        self.interpolateArrays()
        #print(np.min(self.comov_list),np.max(self.comov_list))

        #f, ax = plt.subplots(1, 1, figsize=(15./(1.5*2), 15./(1.5*2)))
        #ax.loglog(self.k_list, self.Pkz_NL(self.k_list,np.min(self.comov_list)))
        #plt.show()

        #print('Ohm_m =', self.cosmo.Omega_m())
        #print('H0 =', self.cosmo.h()*100)

        # Get the cosmic flexion power spectra
        P_kappa_F_list = []
        P_F_list = []
        for l in self.l_list:
            pkapf = self.P_kappa_F(l)
            pf = self.P_F(l)
            P_kappa_F_list.append(pkapf)
            P_F_list.append(pf)
        """
        ## ..Export to a pandas dataframe
        col_list = ['ell', 'P_kappa_F', 'P_F']
        arrs = [self.l_list, P_kappa_F_list, P_F_list]
        dat = {i:arrs[j] for i,j in zip(col_list, range(len(col_list)))}
        out_frame = pd.DataFrame(data = dat, columns = col_list)
        out_frame.to_pickle(self.survey+'/'+self.survey+'_Theory/power_spectra_'+self.survey+'_bin_combo_'+self.bin_combo+'.pkl')
        """
        
        #  First, set the cosmology
        self.cosmology(sigma8 = sigma8+sigma8_error)
        # Next, generate the nonlinear matter power spectrum using CLASS out to kmax_class
        self.P_NL_class()
        # Extrapolate the nonlinear matter power spectrum out to kmax
        #k_P_NL = self.P_NL_extrapolated()
        k_P_NL = self.P_NL_asymptotic()

        # Export k_P_NL(k,z) array to a pickle file
        #with open('k_P_NL.pkl', 'wb') as f:
        #    pickle.dump(k_P_NL, f)
        
        # Next, interpolate all arrays so that they can be turned into callable functions
        self.interpolateArrays()
        #print(np.min(self.comov_list),np.max(self.comov_list))

        #f, ax = plt.subplots(1, 1, figsize=(15./(1.5*2), 15./(1.5*2)))
        #ax.loglog(self.k_list, self.Pkz_NL(self.k_list,np.min(self.comov_list)))
        #plt.show()

        #print('Ohm_m =', self.cosmo.Omega_m())
        #print('H0 =', self.cosmo.h()*100)

        # Get the cosmic flexion power spectra
        P_kappa_F_upper_list = []
        P_F_upper_list = []
        for l in self.l_list:
            pkapf = self.P_kappa_F(l)
            pf = self.P_F(l)
            P_kappa_F_upper_list.append(pkapf)
            P_F_upper_list.append(pf)


        #  First, set the cosmology
        self.cosmology(sigma8 = sigma8-sigma8_error)
        # Next, generate the nonlinear matter power spectrum using CLASS out to kmax_class
        self.P_NL_class()
        # Extrapolate the nonlinear matter power spectrum out to kmax
        #k_P_NL = self.P_NL_extrapolated()
        k_P_NL = self.P_NL_asymptotic()

        # Export k_P_NL(k,z) array to a pickle file
        #with open('k_P_NL.pkl', 'wb') as f:
        #    pickle.dump(k_P_NL, f)
        
        # Next, interpolate all arrays so that they can be turned into callable functions
        self.interpolateArrays()
        #print(np.min(self.comov_list),np.max(self.comov_list))

        #f, ax = plt.subplots(1, 1, figsize=(15./(1.5*2), 15./(1.5*2)))
        #ax.loglog(self.k_list, self.Pkz_NL(self.k_list,np.min(self.comov_list)))
        #plt.show()

        #print('Ohm_m =', self.cosmo.Omega_m())
        #print('H0 =', self.cosmo.h()*100)

        # Get the cosmic flexion power spectra
        P_kappa_F_lower_list = []
        P_F_lower_list = []
        for l in self.l_list:
            pkapf = self.P_kappa_F(l)
            pf = self.P_F(l)
            P_kappa_F_lower_list.append(pkapf)
            P_F_lower_list.append(pf)

        # ..Export to a pandas dataframe
        col_list = ['ell', 'P_kappa_F', 'P_kappa_F_lower', 'P_kappa_F_upper', 'P_F', 'P_F_lower', 'P_F_upper']
        arrs = [self.l_list, P_kappa_F_list, P_kappa_F_lower_list, P_kappa_F_upper_list, P_F_list, P_F_lower_list, P_F_upper_list]
        dat = {i:arrs[j] for i,j in zip(col_list, range(len(col_list)))}
        out_frame = pd.DataFrame(data = dat, columns = col_list)
        out_frame.to_pickle(self.survey+'/'+self.survey+'_Theory/power_spectra_'+self.survey+'_bin_combo_'+self.bin_combo+'_sigma8_varied.pkl')
        """

        # Get q(chi) lists
        q_i_list = []
        q_j_list = []
        for comov in self.comov_list:
            qi = self.q_i(comov)
            qj = self.q_j(comov)
            q_i_list.append(qi)
            q_j_list.append(qj)
        # ..Export to a pandas dataframe
        col_list = ['comov', 'q_i', 'q_j']
        arrs = [self.comov_list, q_i_list, q_j_list]
        dat = {i:arrs[j] for i,j in zip(col_list, range(len(col_list)))}
        out_frame = pd.DataFrame(data = dat, columns = col_list)
        out_frame.to_pickle(self.survey+'/'+self.survey+'_Theory/lensing_efficiency_'+self.survey+'_bin_combo_'+self.bin_combo+'.pkl')


        # Get the cosmic flexion power spectrum
        #P_F_list = []
        #l_list = [1, 10, 100, 1e3, 1e4, 1e5, 1e6]
        #for l in l_list:
        #    pf = self.P_F(l)
        #    P_F_list.append(pf)
        #print(P_F_list)
        #"""
        #f, ax = plt.subplots(1, 1, figsize=(15./(1.5*2), 15./(1.5*2)))
        #ax.loglog(l_list, P_F_list)
        #plt.show()
        
    def cosmology(self, sigma8=0.8111):
        """
        Using the Boltzmann code CLASS, generate the cosmology for 
        General Relativity with Planck18 parameters. We export the nonlinear
        matter power spectrum up to kmax_class.

        cosmology: Planck 18 TT,TE,EE+lowE+lensing (Table 2.)
      
        """
        self.cosmo = Class()
        #zstr=','.join(map(str,self.z_list+self.z_list[-1]+2))
        #z_list = np.append(self.z_list, self.z_list[-1]+2)
        #zstr=','.join(map(str,z_list))
        lcdmpars = {'output': 'mPk',
                    'non linear': 'halofit',
                    'P_k_max_1/Mpc': self.kmax_class,
                    'z_max_pk': self.z_list[-1]+2,
                    'background_verbose': 1, #Info
                    'tau_reio': 0.0544,
                    'omega_cdm': 0.1200,    
                    'sigma8': sigma8,
                    'h': 0.6736,
                    'N_ur': 2.99-1.,  
                    'N_ncdm': 1.,           
                    'm_ncdm': 0.06,       
                    'omega_b': 0.0224, 
                    'n_s': 0.9649, 
                   }
        #'A_s': 2.204e-9, 
        self.cosmo.set(lcdmpars)
        self.cosmo.compute()
        print('Cosmology generated...')

    def P_NL_class(self):
        """
        The nonlinear matter power spectrum generated by the Boltzmann code CLASS.
        Exports an array of the form:
           k  P_NL(k, z_1)  P_NL(k, z_2)  ...  P_NL(k, z_n)
           .       .              .                 .
           .       .              .                 .
           .       .              .                 .
        """
        # Get Hubble constant from cosmology for conversion
        h = self.cosmo.h()

        # First, generate a list of k values in units of h/Mpc
        Nk = 1000
        k_list = np.logspace(np.log10(self.kmin), np.log10(self.kmax_class), Nk)

        # Generate empty 2D array for P_NL(k,z) of dimensions Nk x Nz
        P_NL_list = np.zeros((Nk, self.Nz)) 

        # Populate P_NL(k,z) array. k should have units of 1/Mpc when plugged into P_NL(k,z)
        for z in range(self.Nz):
            for k in range(Nk):
                #P_NL_list[k][z] = self.cosmo.pk(k_list[k]*h, self.z_list[z])#*h**3.
                P_NL_list[k][z] = self.cosmo.pk(k_list[k], self.z_list[z])#*h**3.

        # Combine k_list, and P_NL(k,z)
        k_P_NL = np.column_stack((k_list, P_NL_list))

        #f, ax = plt.subplots(1, 1, figsize=(15./(1.5*2), 15./(1.5*2)))#, sharex='col')
        #ax.loglog(k_list, P_NL_list, color='black') 
        #ax.set_ylabel(r'$\xi_{\mathcal{F}+}(\theta)$')
        #ax.set_xlabel(r'$\theta\,\,{\rm [arcsec]}$')
        #plt.tight_layout()
        #plt.savefig('cosmic_flexion_xi_F_plus.pdf', format='pdf')
        #plt.show()

        #return k_P_NL
        self.k_P_NL_class = k_P_NL

    def P_NL_extrapolated(self):
        """
        Take the nonlinear matter power spectrum that was generated by CLASS, 
        out to a k of kmax_class, and extrapolate to kmax.  This is done for two reasons:
          (i).  We need to go out to essentially arbitrarily large k for purposes of integrating
                the flexion power spectrum from k=0 to k=inf.
          (ii). CLASS is very fast up to kmax_class, but very slow beyond that.  Additionally, the 
                matter power spectrum simply decreases logarithmically beyond kmax_class, so we can
                trivially use linear extrapolation in log space out to arbitrarily large k
        Exports an array of the form:
           k  P_NL(k, z_1)  P_NL(k, z_2)  ...  P_NL(k, z_n)
           .       .              .                 .
           .       .              .                 .
           .       .              .                 .
        """
        # Get k_list and P_NL(k,z) array from class
        k_list_class = self.k_P_NL_class[:,0]
        P_NL_class = self.k_P_NL_class[:,1:]
        N_class = len(k_list_class)
        
        # Generate list of k values for which P_NL will be extrapolated to
        Nk_ext = 1000
        k_list_ext = np.logspace(np.log10(self.kmax_class+1), np.log10(self.kmax), Nk_ext) 
        # Generate empty 2D array for P_NL(k,z) of dimensions Nk_ext x Nz
        P_NL_ext = np.zeros((Nk_ext, self.Nz))
        
        # Combine the two k lists:
        k_list = np.append(k_list_class,k_list_ext)
        #print('k',k_list[N_class+0])
        #print(k_list_ext[0])
        # Combine the two P_NL arrays:
        P_NL = np.row_stack((P_NL_class,P_NL_ext))
        
        for z in range(self.Nz):
            for k in range(Nk_ext):
                P_star = np.log10(P_NL[(N_class+k)-2][z]) + (np.log10(k_list[(N_class+k)]) - np.log10(k_list[(N_class+k)-2]))/(np.log10(k_list[(N_class+k)-1]) - np.log10(k_list[(N_class+k)-2])) * (np.log10(P_NL[(N_class+k)-1][z]) - np.log10(P_NL[(N_class+k)-2][z]))
                P_star = 10**(P_star)
                P_NL[(N_class)+k][z] = P_star 
                
        # Combine k_list, and P_NL(k,z)
        k_P_NL = np.column_stack((k_list, P_NL))
                
        #return k_P_NL
        #return k_P_NL
        self.k_list = k_list
        self.P_NL = P_NL

        ### slope
        slope = (np.log10(P_NL[5][0]) - np.log10(P_NL[0][0]))/(np.log10(k_list[5]) - np.log10(k_list[0]))
        print('ns at z=0 for k < k_peak', slope)
        slope = (np.log10(P_NL[-1][0]) - np.log10(P_NL[-2][0]))/(np.log10(k_list[-1]) - np.log10(k_list[-2]))
        print('ns_prime at z=0 for k >> k_peak is', slope)

    def P_NL_asymptotic(self):
        """
        Take the nonlinear matter power spectrum that was generated by CLASS, 
        out to a k of kmax_class, and extrapolate to kmax.  This is done for two reasons:
          (i).  We need to go out to essentially arbitrarily large k for purposes of integrating
                the flexion power spectrum from k=0 to k=inf.
          (ii). CLASS is very fast up to kmax_class, but very slow beyond that.  Additionally, the 
                matter power spectrum simply follows a power law beyond kmax_class, so we can
                trivially use linear extrapolation in log space out to arbitrarily large k.
                The power law is [arXiv:0901.4576]
                  P_NL \propto k^(ns-4)
                where ns = 0.96 is the index of the primordial power spectrum
        Exports an array of the form:
           k  P_NL(k, z_1)  P_NL(k, z_2)  ...  P_NL(k, z_n)
           .       .              .                 .
           .       .              .                 .
           .       .              .                 .
        """
        # Define primordial power spectrum index
        ns = self.cosmo.n_s()#0.96475
        slope = ns-4
        # Get k_list and P_NL(k,z) array from class
        k_list_class = self.k_P_NL_class[:,0]
        P_NL_class = self.k_P_NL_class[:,1:]
        N_class = len(k_list_class)
        
        # Generate list of k values for which P_NL will be extrapolated to
        Nk_ext = 1000
        k_list_ext = np.logspace(np.log10(self.kmax_class+1), np.log10(self.kmax), Nk_ext) 
        # Generate empty 2D array for P_NL(k,z) of dimensions Nk_ext x Nz
        P_NL_ext = np.zeros((Nk_ext, self.Nz))
        
        # Combine the two k lists:
        k_list = np.append(k_list_class,k_list_ext)
        #print('k',k_list[N_class+0])
        #print(k_list_ext[0])
        # Combine the two P_NL arrays:
        P_NL = np.row_stack((P_NL_class,P_NL_ext))
        
        for z in range(self.Nz):
            for k in range(Nk_ext):
                P_star = np.log10(P_NL[(N_class+k)-1][z]) + (slope)*(np.log10(k_list[(N_class+k)]) - np.log10(k_list[(N_class+k)-1]))
                P_star = 10**(P_star)
                P_NL[(N_class)+k][z] = P_star 
                
        # Combine k_list, and P_NL(k,z)
        k_P_NL = np.column_stack((k_list, P_NL))
                
        #return k_P_NL
        #return k_P_NL
        self.k_list = k_list
        self.P_NL = P_NL

        ### slope
        slope = (np.log10(P_NL[5][0]) - np.log10(P_NL[0][0]))/(np.log10(k_list[5]) - np.log10(k_list[0]))
        print('ns at z=0 for k < k_peak', slope)
        slope = (np.log10(P_NL[-1][0]) - np.log10(P_NL[-2][0]))/(np.log10(k_list[-1]) - np.log10(k_list[-2]))
        print('ns_prime at z=0 for k >> k_peak is', slope)

    def z_to_comov(self):
        """
        Returns comoving distance as a function of redshift
        """
        #comov = Planck15.comoving_distance(self.z_list).value --ignore
        #Planck18 = FlatLambdaCDM(H0=100*self.cosmo.h(), Om0=self.cosmo.)
        #comov = Planck15.comoving_distance(self.z_list).value*self.cosmo.h()
        
        #bg_cosmo = self.cosmo.get_background()
        #comov = bg_cosmo['comov. dist.']
        
        #Planck18 = FlatLambdaCDM(H0=100*self.cosmo.h(), Om0=((self.cosmo.omega_cdm()+self.cosmo.omega_b())/self.cosmo.h()**2.))
        Planck18 = FlatLambdaCDM(H0=100*0.674, Om0=((0.120+0.0224)/0.674**2.))
        comov = Planck18.comoving_distance(self.z_list).value#*self.cosmo.h() #h
        return comov

    def interpolateArrays(self):
        """
        Interpolate all arrays in order to turn them into callable functions.
        Replace redshift with comoving distance in the interpolators.
        """
        # First, get conversion of redshift to comoving distance
        self.comov_list = self.z_to_comov()
        # Nonlinear matter power spectrum P_NL(comov,k)
        self.P_NL_interpolate = interpolate.interp2d(self.comov_list, self.k_list, self.P_NL)
        # n(comov) for bin_i and bin_j
        self.nz_bin_i_interpolate = interpolate.interp1d(self.comov_list, self.nz_bin_i)
        self.nz_bin_j_interpolate = interpolate.interp1d(self.comov_list, self.nz_bin_j)
        
        # Scale factor:
        a_list = 1/(1+self.z_list)
        self.a_interpolate = interpolate.interp1d(self.comov_list, a_list)
        

    def Pkz_NL(self, k, comov):
        return self.P_NL_interpolate(comov, k)

    def n_i(self, comov):
        return self.nz_bin_i_interpolate(comov)

    def n_j(self, comov):
        return self.nz_bin_j_interpolate(comov)

    def a(self, comov):
        return self.a_interpolate(comov)
    
    def q_i(self, comov):
        """
        Lensing efficiency function for bin_i
        """
        def integrand(comov_prime):
            return self.n_i(comov_prime)*(comov_prime-comov)/comov_prime
        integral = integrate.quad(integrand, comov, np.max(self.comov_list))[0]

        q_i = (3/2)*self.cosmo.Omega_m()*(100*self.cosmo.h()/3e5)**2. * (comov/self.a(comov)) * integral

        return q_i

    def q_j(self, comov):
        """
        Lensing efficiency function for bin_j
        """
        def integrand(comov_prime):
            return self.n_j(comov_prime)*(comov_prime-comov)/comov_prime
        integral = integrate.quad(integrand, comov, np.max(self.comov_list))[0]

        q_j = (3/2)*self.cosmo.Omega_m()*(100*self.cosmo.h()/3e5)**2. * (comov/self.a(comov)) * integral

        return q_j

    def P_F(self, ell):
        """
        Cosmic flexion power spectrum as a function of angular wavenumber ell
        """
        def integrand(comov):
            return (self.q_i(comov)*self.q_j(comov)/comov**2.)*self.Pkz_NL((ell)/comov,comov)
        integral = integrate.quad(integrand, np.min(self.comov_list), np.max(self.comov_list))
        #print(integral)
        P_F = ell**2.*integral[0]
     
        return P_F

    def P_kappa_F(self, ell):
        """
        Cosmic shear-flexion cross power spectrum as a function of angular wavenumber ell
        """
        def integrand(comov):
            return (self.q_i(comov)*self.q_j(comov)/comov**2.)*self.Pkz_NL((ell)/comov,comov)
        integral = integrate.quad(integrand, np.min(self.comov_list), np.max(self.comov_list))
        #print(integral)
        P_kap_F = ell*integral[0]
     
        return P_kap_F




"""
# Simple test to check a few things.  First, let's just check that this extrapolates properly.
# Pick two redshifts (z=0 and z=0.01) to look at
k_P_NL = CoFlexPower([0,0.01]).getPower()

f, ax = plt.subplots(1, 1, figsize=(15./(1.5*2), 15./(1.5*2)))
ax.loglog(k_P_NL[:,0], k_P_NL[:,1])
plt.show()

#Next, do an interpolation test

#print('k,z,P_NL 0 before interpolation', k_P_NL[:,0][0], 0, 

P_NL_interpolate = interpolate.interp2d([0,0.01], k_P_NL[:,0], k_P_NL[:,1:])

def P_NL(z,k):
    return P_NL_interpolate(z,k)

print(P_NL(0.0001,1e-4))

print(k_P_NL[:,1][0])
"""


#z_list = np.linspace(0,4,400)
#MatterPower(z_list).getPower()
