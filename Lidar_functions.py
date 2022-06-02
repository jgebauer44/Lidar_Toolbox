#!/opt/local/anaconda3/bin/python
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 11:16:18 2021

@author: jgebauer
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate
from scipy.optimize import leastsq

##############################################################################
# This is a class created for VAD data
##############################################################################
class VAD:
    def __init__(self,u,v,w,speed,wdir,du,dv,dw,z,residual,time):
       self.u = np.array(u)
       self.v = np.array(v)
       self.w = np.array(w)
       self.speed = np.array(speed)
       self.wdir = np.array(wdir)
       self.du = du
       self.dv = dv
       self.dw = dw
       self.z = z
       self.residual = np.array(residual)
       self.time = np.array(time)

##############################################################################
# This is a class created for gridded RHI data
##############################################################################

class gridded_RHI:
    def __init__(self,field,x,z,dx,offset,grid_el,grid_range,time):
        self.field = np.array(field)
        self.x = np.array(x)
        self.z = np.array(z)
        self.dx = dx
        self.x_offset = offset[0]
        self.z_offset = offset[1]
        self.time = np.array(time)
        self.grid_el = grid_el
        self.grid_range = grid_range

##############################################################################
# This is a class created for a vertical profile from a lidar RHI or multiple
# VADs
##############################################################################

class vertical_vr:
    def __init__(self,field,x_loc,y_loc,z,dz,offset,z_el,z_range,az,time):
        self.field = np.array(field)
        self.x = x_loc
        self.y = y_loc
        self.z = np.array(z)
        self.dz = dz
        self.x_offset = offset[0]
        self.y_offset = offset[1]
        self.z_offset = offset[2]
        self.time = np.array(time)
        self.azimuth = az
        self.elevation = np.array(z_el)
        self.range = np.array(z_range)
 
##############################################################################
# This function calculates VAD wind profiles using the technique shown in
# Newsom et al. (2019). This function can calculate one VAD profile or a series
# of VAD profiles depending on the radial velocity input
##############################################################################

def ARM_VAD(radial_vel,ranges,el,az,time=None,missing=None):
    
    if (time is None) & (len(radial_vel.shape) == 2):
        times = 1
        time = [0]
        vr = np.array([radial_vel])
    elif (time is None) & (len(radial_vel.shape) == 3):
        time = np.arange(radial_vel.shape[0])
        vr = np.copy(radial_vel)
        times = len(time)
    else:
        times = len(time)
        vr = np.copy(radial_vel)
        
    if missing is not None:
        vr[vr==missing] = np.nan
    
    x = ranges[None,:]*np.cos(np.radians(el))*np.sin(np.radians(az[:,None]))
    y = ranges[None,:]*np.cos(np.radians(el))*np.cos(np.radians(az[:,None]))
    z = ranges*np.sin(np.radians(el))
    
    u = []
    v = []
    w = []
    residual = []
    speed = []
    wdir = []
    
    for j in range(times):
        temp_u = np.ones(len(ranges))*np.nan
        temp_v = np.ones(len(ranges))*np.nan
        temp_w = np.ones(len(ranges))*np.nan
        
        for i in range(len(ranges)):
            foo = np.where(~np.isnan(vr[j,:,i]))[0]

            if len(foo) == 0:
                temp_u[i] = np.nan
                temp_v[i] = np.nan
                temp_w[i] = np.nan
                continue

            A11 = (np.cos(np.deg2rad(el))**2) * np.sum(np.sin(np.deg2rad(az[foo]))**2)
            A12 = (np.cos(np.deg2rad(el))**2) * np.sum(np.sin(np.deg2rad(az[foo])) * np.cos(np.deg2rad(az[foo])))
            A13 = (np.cos(np.deg2rad(el))*np.sin(np.deg2rad(el))) * np.sum(np.sin(np.deg2rad(az[foo])))
            A22 = (np.cos(np.deg2rad(el))**2) * np.sum(np.cos(np.deg2rad(az[foo]))**2)
            A23 = (np.cos(np.deg2rad(el))*np.sin(np.deg2rad(el))) * np.sum(np.cos(np.deg2rad(az[foo])))
            A33 = len(az[foo]) * (np.sin(np.deg2rad(el))**2)

            A = np.array([[A11,A12,A13],[A12,A22,A23],[A13,A23,A33]])
            invA = np.linalg.inv(A)
    
            du = invA[0,0]
            dv = invA[1,1]
            dw = invA[2,2]
    
            b1 = np.cos(np.deg2rad(el)) * np.sum(vr[j,foo,i] * np.sin(np.deg2rad(az[foo])))
            b2 = np.cos(np.deg2rad(el)) * np.sum(vr[j,foo,i] * np.cos(np.deg2rad(az[foo])))
            b3 = np.sin(np.deg2rad(el)) * np.sum(vr[j,foo,i])
    
            b = np.array([b1,b2,b3])
    
            temp = invA.dot(b)
            temp_u[i] = temp[0]
            temp_v[i] = temp[1]
            temp_w[i] = temp[2]
    
        u.append(np.copy(temp_u))
        v.append(np.copy(temp_v))
        w.append(np.copy(temp_w))
    
        residual.append(np.sqrt(np.nanmean(((((temp_u*x)+(temp_v*y)+((temp_w*z)[None,:]))/np.sqrt(x**2+y**2+z**2))-vr[j])**2,axis = 0)))
        speed.append(np.sqrt(temp_u**2 + temp_v**2))
        temp_wdir = 270 - np.rad2deg(np.arctan2(temp_v,temp_u))

        foo = np.where(temp_wdir >= 360)[0]
        temp_wdir[foo] -= 360

        wdir.append(temp_wdir)

    return VAD(u,v,w,speed,wdir,du,dv,dw,z,residual,time)
            
    
##############################################################################
# This function will plot wind profiles from VAD objects
##############################################################################

def plot_VAD(vad,filename,plot_time = None, title=None):
    
    if plot_time is None:
        plot_time = len(vad.time) - 1
        start = 0
    else:
        start = plot_time
    
    for i in range(start,plot_time+1):
        plt.figure(figsize=(12,8))
        ax = plt.subplot(141)
        plt.plot(vad.speed[i],vad.z)
        ax.set_xlim(0,30)
        ax.set_ylim([0,np.max(vad.z)])
        ax.set_xlabel('Wind Speed [m/s]')
        ax.set_ylabel('Height [m]')
 
        ax = plt.subplot(142)
        plt.plot(vad.wdir[i],vad.z)
        ax.set_xlim([0,360])
        ax.set_ylim([0,np.max(vad.z)])
        ax.set_xlabel('Wind Direction')

        ax = plt.subplot(143)
        plt.plot(vad.w[i],vad.z)
        ax.set_xlim(-5,5)
        ax.set_ylim([0,np.max(vad.z)])
        ax.set_xlabel('Vertical Velocity [m/s]')
    
        ax = plt.subplot(144)
        plt.plot(vad.residual[i],vad.z)
        ax.set_xlim(0,10)
        ax.set_ylim([0,np.max(vad.z)])
        ax.set_xlabel('Residual')

        if title is not None:
            plt.title(title)
    
        plt.tight_layout()
        
        if ((start == 0) & (len(vad.time) > 1)):
            if os.path.isdir(filename):
                plt.savefig(filename + str(vad.time[i]))
            else:
                os.mkdir(filename)
                plt.savefig(filename + str(vad.time[i]))
        else:
            plt.savefig(filename)
        
        plt.close()

##############################################################################
# This function will take RHI scans and will put them onto a 2-D cartesian grid
# using linear interpolation
##############################################################################

def grid_rhi(field,elevation,ranges,dims,dx,offset=None,
             time=None,missing=None):
    
    if len(dims) != 2:
        raise IOError('Dims must be a 2 length tuple')
    
    if offset is not None:
        if len(offset) != 2:
            raise IOError('If offset is specified it must be 2 length tuple')
    else:
        offset = (0,0)
        
    if (time is None) & (len(field.shape) == 2):
        times = 1
        time = [0]
        raw = np.array([field])
        el = np.array([elevation])
    elif (time is None) & (len(field.shape) == 3):
        time = np.arange(field.shape[0])
        el = np.copy(elevation)
        raw = np.copy(field)
        times = len(time)
    else:
        times = len(time)
        raw = np.copy(field)
        el = np.copy(elevation)
        
    if missing is not None:
        raw[raw==missing] = np.nan
        
    x = ranges[None,:] * np.cos(np.deg2rad(180-el))[:,:,None] + offset[0]
    z = ranges[None,:] * np.sin(np.deg2rad(el))[:,:,None] + offset[1]
    
    grid_x, grid_z = np.meshgrid(np.arange(dims[0][0],dims[0][1]+1,dx),np.arange(dims[1][0],dims[1][1]+1,dx))
    
    grid_range = np.sqrt((grid_x-offset[0])**2 + (grid_z-offset[1])**2)
    grid_el = 180 - np.rad2deg(np.arctan2(grid_z-offset[1],grid_x-offset[0]))

    grid_field = []
    for i in range(times):
        foo = np.where(~np.isnan(x[i]))
        
        grid_field.append(scipy.interpolate.griddata((x[i,foo[0],foo[1]],z[i,foo[0],foo[1]]),
                                                     raw[i,foo[0],foo[1]],(grid_x,grid_z)))
    
    return gridded_RHI(grid_field,grid_x,grid_z,dx,offset,grid_el,grid_range,time)

##############################################################################
# This function calculates a coplanar wind field from two gridded RHIs
##############################################################################

def coplanar_analysis(vr1,vr2,el1,el2,az):
    
    u = np.ones(vr1.shape)*np.nan
    w = np.ones(vr1.shape)*np.nan
    
    for i in range(vr1.shape[0]):
        for j in range(vr1.shape[1]):
            
            if ((~np.isnan(vr1[i,j])) & (~np.isnan(vr2[i,j]))):
                M = np.array([[np.sin(np.deg2rad(az))*np.cos(np.deg2rad(el1[i,j])),np.sin(np.deg2rad(el1[i,j]))],
                              [np.sin(np.deg2rad(az))*np.cos(np.deg2rad(el2[i,j])),np.sin(np.deg2rad(el2[i,j]))]])
    
                U = np.linalg.solve(M,np.array([vr1[i,j],vr2[i,j]]))
                u[i,j] = U[0]
                w[i,j] = U[1]
    
    return u, w

##############################################################################
# The function calculates the vr-variance from a timeseries of scans
##############################################################################

def vr_variance(field,time,t_avg,axis=0):
    
    t_avg = t_avg*60
    start = 0
    yo = np.where(time < (time[0]+t_avg))[0]
    end = yo[-1]+1
    var = []
    time_avg = []
    while end < len(time):
        var.append(np.nanvar(field[start:end,:,:],axis=0))
        time_avg.append(np.nanmean(time[start:end]))
        start = end
        yo = np.where(time < (time[start]+t_avg))[0]
        end = yo[-1]+1
    
    return np.array(var),np.array(time_avg)

##############################################################################
# This function puts vr observations from a RHI onto a vertical grid at one
# location to be used for virtual tower calculations
##############################################################################

def rhi_vertical_profile(field,elevation,azimuth,ranges,heights,dz,loc,offset=None,
                         time=None,missing=None):
    
    if len(loc) != 2:
        raise IOError('Dims must be a 2 length tuple')
    
    if offset is not None:
        if len(offset) != 3:
            raise IOError('If offset is specified it must be 3 length tuple')
    else:
        offset = (0,0,0)
        
    if (time is None) & (len(field.shape) == 2):
        times = 1
        time = [0]
        raw = np.array([field])
        el = np.array([elevation])
    elif (time is None) & (len(field.shape) == 3):
        time = np.arange(field.shape[0])
        el = np.copy(elevation)
        raw = np.copy(field)
        times = len(time)
    else:
        times = len(time)
        raw = np.copy(field)
        el = np.copy(elevation)
    
    if missing is not None:
        raw[raw==missing] = np.nan
        
    x = ranges[None,:] * np.cos(np.deg2rad(el))[:,:,None]*np.sin(np.deg2rad(azimuth)) + offset[0]
    y = ranges[None,:] * np.cos(np.deg2rad(el))[:,:,None]*np.cos(np.deg2rad(azimuth)) + offset[1]
    z = ranges[None,:] * np.sin(np.deg2rad(el))[:,:,None] + offset[2]
    
    r = np.sqrt(x**2 + y**2)

    z_interp = np.arange(heights[0],heights[1],dz)
    
    z_ranges = np.sqrt((loc[0]-offset[0])**2 + (loc[1]-offset[1])**2+(z_interp-offset[2])**2)
    z_el = np.rad2deg(np.arctan2(z_interp-offset[2],np.sqrt((loc[0]-offset[0])**2+(loc[1]-offset[1])**2)))
    
    grid_field = []
    grid_x,grid_z = np.meshgrid(np.array(np.sqrt(loc[0]**2 + loc[1]**2)),z_interp)
    for i in range(times):
        grid_field.append(scipy.interpolate.griddata((r[i].ravel(),z[i].ravel()),
                                                     raw[i].ravel(),(grid_x,grid_z))[:,0])
    
    return vertical_vr(grid_field,loc[0],loc[1],z_interp,dz,offset,z_el,z_ranges,azimuth,time)
    
##############################################################################
# This function calculates the wind components for a virtual towers
##############################################################################

def virtual_tower(vr,elevation,azimuth,height,uncertainty = 0.45):
    
    if len(vr) == 2:
        u = []
        v = []
        u_uncertainty = []
        v_uncertainty = []
        el1 = np.deg2rad(elevation[0])
        el2 = np.deg2rad(elevation[1])
        az1 = np.deg2rad(azimuth[0])
        az2 = np.deg2rad(azimuth[1])
        for i in range(len(height)):
            if ((np.isnan(vr[0][i]) | np.isnan(vr[1][i]))):
                u.append(np.nan)
                v.append(np.nan)
                
                M = np.array([[np.sin(az1)*np.cos(el1[i]),np.cos(az1)*np.cos(el1[i])],
                              [np.sin(az2)*np.cos(el2[i]),np.cos(az2)*np.cos(el2[i])]])
                invM = np.linalg.inv(M)
                temp_u = np.sqrt((invM[0,0]**2)*(uncertainty**2) + (invM[0,1]**2)*(uncertainty**2))
                temp_v = np.sqrt((invM[1,0]**2)*(uncertainty**2) + (invM[1,1]**2)*(uncertainty**2))
                u_uncertainty.append(temp_u)
                v_uncertainty.append(temp_v)

            else:
                M = np.array([[np.sin(az1)*np.cos(el1[i]),np.cos(az1)*np.cos(el1[i])],
                              [np.sin(az2)*np.cos(el2[i]),np.cos(az2)*np.cos(el2[i])]])
                temp = np.linalg.solve(M,np.array([vr[0][i],vr[1][i]]))
                u.append(np.copy(temp[0]))
                v.append(np.copy(temp[1]))
                
                invM = np.linalg.inv(M)
                temp_u = np.sqrt((invM[0,0]**2)*(uncertainty**2) + (invM[0,1]**2)*(uncertainty**2))
                temp_v = np.sqrt((invM[1,0]**2)*(uncertainty**2) + (invM[1,1]**2)*(uncertainty**2))
                u_uncertainty.append(temp_u)
                v_uncertainty.append(temp_v)
                
        return np.array([u,v]),np.array([u_uncertainty,v_uncertainty])
                              
    elif len(vr) == 3:
        u = []
        v = []
        w = []
        u_uncertainty = []
        v_uncertainty = []
        w_uncertainty = []
        el1 = np.deg2rad(elevation[0])
        el2 = np.deg2rad(elevation[1])
        el3 = np.deg2rad(elevation[2])
        az1 = np.deg2rad(azimuth[0])
        az2 = np.deg2rad(azimuth[1])
        az3 = np.deg2rad(azimuth[2])
        for i in range(len(height)):
            if ((np.isnan(vr[0][i]) | np.isnan(vr[1][i]))):
                u.append(np.nan)
                v.append(np.nan)
                
                M = np.array([[np.sin(az1)*np.cos(el1[i]),np.cos(az1)*np.cos(el1[i]),np.sin(el1)],
                              [np.sin(az2)*np.cos(el2[i]),np.cos(az2)*np.cos(el2[i]),np.sin(el2)],
                              [np.sin(az3)*np.cos(el3[i]),np.cos(az3)*np.cos(el3[i]),np.sin(el3)]])
                invM = np.linalg.inv(M)
                temp_u = np.sqrt((invM[0,0]**2)*(uncertainty**2) + (invM[0,1]**2)*(uncertainty**2) + (invM[0,2]**2)*(uncertainty**2))
                temp_v = np.sqrt((invM[1,0]**2)*(uncertainty**2) + (invM[1,1]**2)*(uncertainty**2) + (invM[1,2]**2)*(uncertainty**2))
                temp_w = np.sqrt((invM[2,0]**2)*(uncertainty**2) + (invM[2,1]**2)*(uncertainty**2) + (invM[2,2]**2)*(uncertainty**2))
                
                u_uncertainty.append(temp_u)
                v_uncertainty.append(temp_v)
                w_uncertainty.append(temp_w)
            else:
                M = np.array([[np.sin(az1)*np.cos(el1[i]),np.cos(az1)*np.cos(el1[i]),np.sin(el1)],
                              [np.sin(az2)*np.cos(el2[i]),np.cos(az2)*np.cos(el2[i]),np.sin(el2)],
                              [np.sin(az3)*np.cos(el3[i]),np.cos(az3)*np.cos(el3[i]),np.sin(el3)]])
                temp = np.linalg.solve(M,np.array([vr[0][i],vr[1][i],vr[2][i]]))
                u.append(np.copy(temp[0]))
                v.append(np.copy(temp[1]))
                w.append(np.copy(temp[2]))
                
                invM = np.linalg.inv(M)
                temp_u = np.sqrt((invM[0,0]**2)*(uncertainty**2) + (invM[0,1]**2)*(uncertainty**2) + (invM[0,2]**2)*(uncertainty**2))
                temp_v = np.sqrt((invM[1,0]**2)*(uncertainty**2) + (invM[1,1]**2)*(uncertainty**2) + (invM[1,2]**2)*(uncertainty**2))
                temp_w = np.sqrt((invM[2,0]**2)*(uncertainty**2) + (invM[2,1]**2)*(uncertainty**2) + (invM[2,2]**2)*(uncertainty**2))
                
                u_uncertainty.append(temp_u)
                v_uncertainty.append(temp_v)
                w_uncertainty.append(temp_w)
                
        return np.array([u,v,w]), np.array([u_uncertainty,v_uncertainty,w_uncertainty])
    else:
        print('Input needs to be a length 2 or 3 tuple')
        return np.nan

##############################################################################
# This function performs a Lenshow correction to lidar data and calculates
# vertical velocity variance. This function was originally written by Tyler
# Bell at CIMMS in Norman, OK.
##############################################################################

def lenshow(x, freq=1, tau_min=3, tau_max=12, plot=False):
    """
    Reads in a timeseries. Freq is in Hz. Default taus are from avg values from Bonin Dissertation (2015)
    Returns avg w'**2 and avg error'**2
    """
    # Find the perturbation of x
    mean = np.mean(x)
    prime = x - mean
    # Get the autocovariance 
    acorr, lags = xcorr(prime, prime)
    var = np.var(prime)
    acov = acorr# * var
    # Extract lags > 0
    lags = lags[int(len(lags)/2):] * freq
    acov = acov[int(len(acov)/2):]
    # Define the start and end lags
    lag_start = int(tau_min / freq)
    lag_end = int(tau_max / freq)
    # Fit the structure function
    fit_funct = lambda p, t: p[0] - p[1]*t**(2./3.) 
    err_funct = lambda p, t, y: fit_funct(p, t) - y
    p1, success = leastsq(err_funct, [1, .001], args=(lags[lag_start:lag_end], acov[lag_start:lag_end]))
    if plot:
        new_lags = np.arange(tau_min, tau_max)
        plt.plot(lags, acov)
        plt.plot(new_lags, fit_funct(p1, new_lags), 'gX')
        plt.plot(0, fit_funct(p1, 0), 'gX')
        plt.xlim(0, tau_max+20)
        plt.xlabel("Lag [s]")
        plt.ylabel("$M_{11} [m^2s^{-2}$]")
    return p1[0], acov[0] - p1[0]

##############################################################################
# This is a modified version of the Lenshow correction that adaptively selects
# taus based on the data. This function was originally written by Tyler Bell at
# CIMMS in Norman, OK.
###############################################################################
    
def lenshow_bonin(x, tau_min=1, tint_first_guess=3, freq=1, max_iter=100, plot=False):
    # Find the perturbation of x
    mean = np.mean(x)
    prime = x - mean
    # Get the autocovariance 
    acorr, lags = xcorr(prime, prime)
    var = np.var(prime)
    acov = acorr #* var
    # Extract lags > 0
    lags = lags[int(len(lags)/2):] * freq
    acov = acov[int(len(acov)/2):]
    # Define the start and end lags
    lag_start = int(tau_min / freq)
    lag_end = int((tau_min+3) / freq)
    # Fit the structure function
    fit_funct = lambda p, t: p[0] - p[1]*t**(2./3.) 
    err_funct = lambda p, t, y: fit_funct(p, t) - y
    # Iterate to find t_int
    last_tint = (tau_min+3)
    i = 0
    p1, success = leastsq(err_funct, [.10, .001], args=(lags[lag_start:lag_end], acov[lag_start:lag_end]))
    tint = calc_tint(p1[0], freq, acov, lags)
    while np.abs(last_tint - tint) > 1.:
        if i >= max_iter:
            return None
        else:
            i += 1
            last_tint = tint
        p1, success = leastsq(err_funct, [.10, .001], args=(lags[lag_start:lag_end], acov[lag_start:lag_end]))
        tint = calc_tint(p1[0], freq, acov, lags)
    # Find the time where M11(t) = M11(0)/2
    ind = np.min(np.where(acov <= acov[0]/2))
    # Determine what tau to use
    tau_max = np.min([tint/2., lags[ind]])
    # Do the process
    lag_end = int(tau_max / freq)
    if lag_start+1 >= lag_end:
        lag_end = lag_start + 2
#     print lag_start, lag_end
    p1, success = leastsq(err_funct, [.10, .001], args=(lags[lag_start:lag_end], acov[lag_start:lag_end]))
    if plot:
        new_lags = np.arange(tau_min, tau_max)
        plt.plot(lags, acov, 'k')
        plt.plot(new_lags, fit_funct(p1, new_lags), 'rX', label='Adaptive')
        plt.plot(0, fit_funct(p1, 0), 'rX')
        plt.xlim(0, tau_max+20)
        plt.xlabel("Lag [s]")
        plt.ylabel("$M_{11} [m^2s^{-2}$]")
    return p1[0], np.abs(acov[0] - p1[0]), tau_max

##############################################################################
# This function is used in the lenshow_bonin fuction
##############################################################################

def calc_tint(var, freq, acov, lags):
    ind = np.min(np.where(acov < 0))
    return freq**-1. + 1./var * sum(acov[1:ind] / freq)

##############################################################################
# This function calculates the lag correlation of a time series.
##############################################################################

def xcorr(y1,y2):
    
    if len(y1) != len(y2):
        raise ValueError('The lenghts of the inputs should be the same')
    
    corr = np.correlate(y1,y2,mode='full')
    unbiased_size = np.correlate(np.ones(len(y1)),np.ones(len(y1)),mode='full')
    corr = corr/unbiased_size
    
    maxlags = len(y1)-1
    lags = np.arange(-maxlags,maxlags + 1)
    
    return corr,lags

##############################################################################
# This function will work with LidarSim data
##############################################################################

def process_LidarSim_scan(scan,scantype,elevation,azimuth,ranges,time):
    
    if scantype == 'vad':
        el = np.nanmean(elevation)
        vad = ARM_VAD(scan,ranges,el,azimuth,time)
        
        return vad
    
    if scantype = 'DBS':
        
        el = np.nanmean(elevation)
        dbs = DBS()
    else:
        print('Not a valid scan type')
        return np.nan
        
        
        
        
    
