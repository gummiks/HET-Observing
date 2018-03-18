# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 19:36:52 2016

@authors: Ryan Terrien, Gudmundur Stefansson
"""

from numpy import *
from matplotlib.pyplot import *
import pandas as pd
from scipy.interpolate import griddata
import glob
from astropy.io import fits
import os
from lmfit import minimize, Parameters, Parameter, report_fit
from lmfit.models import LinearModel
import scipy.signal
import scipy.constants as const
import astropy
import astropy.coordinates
import astroplan
import astropy.units as u
import astroplan.plots
from skyfield.api import load, Star, Angle
#from tqdm import tqdm_notebook as tqdm
from tqdm import tqdm
import het_helper_functions
import het_config

class HETObservability:
    """
    A class to check the observability for targets when observing with HET.

    INPUT:

    OUTPUT:
        het_helper_functions.arange_time('2018-02-25 05:00:00','2018-03-25 05:00:00',form="iso")

    EXAMPLE:
        HTO = het_obs.HETObservability('2018-02-25 05:00:00','2018-03-05 05:00:00')

        A = het_helper_functions.AngleWithProperMotion()
        cc_new = [A.radec_with_proper_motion(df_cut["RA_SIMBAD"].values[i],
                                             df_cut["DEC_SIMBAD"].values[i],
                                             df_cut["PMRA"].values[i],
                                             df_cut["PMDEC"].values[i]) for i in range(len(df_cut))]
        # We want the number of observable days
        observable_days = [len(HTO.get_observable_days(c,verbose=True)) for c in cc_new] 
        df_obs = df_cut[np.array(observable_days)>0]
    """
    def __init__(self,startdate="",stopdate="",twilight_angle=-18.,verbose=True):
        """
        EXAMPLE:
            startdate = '2018-02-25 05:00:00'
            stopdate = '2018-03-05 05:00:00' 
        """
        print("Using Twilight = {}".format(twilight_angle))
        self.hetloc = astropy.coordinates.EarthLocation.of_site('McDonald Observatory')
        self.observer = astroplan.Observer(location=self.hetloc)

        #################################
        all_times = het_helper_functions.arange_time(startdate,stopdate,form=None) # array of astropy.time.Time()
        self.num_nights = len(all_times)
        jd_arr = [i.jd for i in all_times]
        hori = astropy.coordinates.Angle(twilight_angle,unit=u.degree) # have used -12
        print("Crating array of {} nights to check observability".format(self.num_nights))

        #################################
        print("Calculating Sunrise/Sunset Times")
        sunrise_times, sunset_times, self.times_night_all_lst = [], [], []
        for i in tqdm(all_times):
            sunrise_times.append(astropy.time.Time(self.observer.sun_rise_time(i,which='nearest',horizon=hori),
                location=self.hetloc))
            sunset_times.append(astropy.time.Time(self.observer.sun_set_time(i,which='nearest',horizon=hori),
                location=self.hetloc))
        self.df_time = pd.DataFrame({'jd':jd_arr,'time':all_times,'sunset_time':sunset_times,
            'sunrise_time':sunrise_times})

        #################################
        print("Calculating Local Sidereal Times")
        for i in tqdm(self.df_time.index):
            jd_1 = self.df_time.loc[i,'sunset_time'].jd
            jd_2 = self.df_time.loc[i,'sunrise_time'].jd
            jds_night = linspace(jd_1,jd_2,100)
            self.times_night_all_lst.append([astropy.time.Time(i,format='jd',
                location=self.hetloc).sidereal_time('mean').degree for i in jds_night])
        
        #################################
        # Setting observability limits
        decs = linspace(-90.,90.,10000)#dindgen(1d5)/1d5*180d - 90d ;declinations to try
        decs_rad = radians(decs) #del_rad = del * !dtor

        phi = self.hetloc.latitude.radian
        alt1 = radians(het_config.HET_MAX_ALT)
        alt2 = radians(het_config.HET_MIN_ALT)
        
        h1 = degrees(arccos( (sin(alt1) - sin(decs_rad)*sin(phi)) / (cos(decs_rad)*cos(phi)) ))
        h2 = degrees(arccos( (sin(alt2) - sin(decs_rad)*sin(phi)) / (cos(decs_rad)*cos(phi)) ))
        
        # The following structure shows the observability limits for each declination value
        # where limits are degrees from zenith (hour angle), and are symmetric across zenith 
        # (ie negative limits also apply)
        self.obs_limits = {'h1':h1,'h2':h2,'decs':decs,'alt1':alt1,'alt2':alt2}
        print("Finished loading observability limits for all nights")
    
    def get_observable_days(self,target_coord,verbose=True):
        """
        Returns the observable days a target at HET for dates given in start/stoptimes at the initialization of the object.

        INPUT:
            target_coord

        OUTPUT:
            The number of days the target is observable

        NOTES:
            Only returns the approximate day that it is observable (i.e., not exactly when).

        EXAMPLE:
            HTO = het_obs.HETObservability('2018-02-25 05:00:00','2018-03-25 05:00:00')

            A = het_helper_functions.AngleWithProperMotion()
            cc_new = [A.radec_with_proper_motion(df_cut["RA_SIMBAD"].values[i],
                                                 df_cut["DEC_SIMBAD"].values[i],
                                                 df_cut["PMRA"].values[i],
                                                 df_cut["PMDEC"].values[i]) for i in range(len(df_cut))]
            # We want the number of observable days
            observable_days = [len(HTO.get_observable_days(c,verbose=True)) for c in cc_new] 
            df_obs = df_cut[np.array(observable_days)>0]
        """
        observable_days = []

        # Loop through all of the dates
        for i_day in self.df_time.index:
            times_night_lst = self.times_night_all_lst[i_day]
            has = (array(times_night_lst) - target_coord.ra.degree) % 360.
            has[has > 180.] -= 360.
            ha_lim_1 = interp(target_coord.dec.degree,self.obs_limits['decs'],self.obs_limits['h1'])
            ha_lim_2 = interp(target_coord.dec.degree,self.obs_limits['decs'],self.obs_limits['h2'])

            # have to catch when tracks are not split
            decs_split_i = nonzero(isfinite(self.obs_limits['h1']))
            dec_low = amin(self.obs_limits['decs'][decs_split_i])
            dec_high = amax(self.obs_limits['decs'][decs_split_i])

            if (target_coord.dec.degree <= dec_low) | (target_coord.dec.degree >= dec_high):
                tt_1 = abs(has) < ha_lim_2
            else:
                tt_1 = (abs(has) > ha_lim_1) & (abs(has) < ha_lim_2)
            if any(tt_1):
                # Target is observable, append the date in iso format
                observable_days.append(self.df_time.time.values[i_day].iso)

        if verbose: print("Target is observable on {} days out of {}".format(len(observable_days),self.num_nights))

        return observable_days
    
    
def print_het_targs(inp):
    '''
    This prints the targets for the phase 1 proposal
    :param inp: pandas dataframe with name, ra(string), de(string), Vmag
    :return:
    '''
    os = []
    for i in inp.index:
        o1 = []
        o1.append('\ObjName{{{}}}'.format(inp.loc[i,'name']).replace('_',' '))
        o1.append('\NumberofObjects{1}')
        o1.append('\\ra{{{}}}'.format(inp.loc[i,'ra_s']))
        o1.append('\\dec{{{}}}'.format(inp.loc[i,'de_s']))
        o1.append('\magnitude{{{:2.3}}}'.format(inp.loc[i,'V']))
        o1.append('\Filter{V}')
        o1.append('\AcquisitionMethod{Finder Chart}')
        o2 = '\n'.join(o1)
        os.append(o2)
    return '\n\n'.join(os)


class FinderChartMaker:
    '''
    This object makes the finder charts in a format friendly to the HET TOs
    The self.t current time parameter should be updated if there are high-PM stars
    '''
    def __init__(self):
        planets = load('de421.bsp')
        ts = load.timescale()
        self.earth = planets['earth']
        self.t = ts.utc(2018.0)
        # these two epochs work for most of the finder charts
        self.t_1989 = ts.utc(1990) # was 1989, but stars were offset bc of varying epoch
        self.t_2000 = ts.utc(2000)

    def make_chart(self,ra,de,pmra,pmde,name,outdir = 'findercharts_2/'):
        '''

        :param ra: ra in deg
        :param de: dec in deg
        :param pmra: pmra in mas/yr
        :param pmde: pmde in mas/yr
        :param name:
        :param outdir: directory for plots
        :return:
        '''
        ra_ang = Angle(degrees=ra)
        de_ang = Angle(degrees=de)
        star_obj = Star(ra=ra_ang,dec=de_ang,ra_mas_per_year=pmra,dec_mas_per_year=pmde)
        astrometric = self.earth.at(self.t).observe(star_obj)
        astrometric_1989 = self.earth.at(self.t_1989).observe(star_obj)
        ra_out,dec_out,dist_out = astrometric.radec()
        ra_out1,dec_out1,dist_out1 = astrometric_1989.radec()
        ra_new = ra_out.to(u.degree).value
        de_new = dec_out.to(u.degree).value
        ra_old = ra_out1.to(u.degree).value
        de_old = dec_out1.to(u.degree).value
        raline = [ra_old,ra_new]
        deline = [de_old,de_new]
        cc = astropy.coordinates.SkyCoord(ra,de,unit=(u.degree,u.degree))
        cc_new = astropy.coordinates.SkyCoord(ra_new,de_new,unit=(u.degree,u.degree))
        cc_old = astropy.coordinates.SkyCoord(ra_old,de_old,unit=(u.degree,u.degree))
        fig=figure(figsize=(10,12))
        fovrad = astropy.coordinates.Angle('0d4m0s')
        if isfinite(pmra):
            #print 'pm finite'
            if max(pmra,pmde) > 1000:
                fovrad = astropy.coordinates.Angle('0d8m0s')
            if max(pmra,pmde) > 5000:
                print 'big radius'
                fovrad = astropy.coordinates.Angle('0d16m0s')
        oo,oh = astroplan.plots.plot_finder_image(cc_old,reticle=True,survey='DSS2 Red',fov_radius=fovrad)
        oo.set_autoscale_on(False)
        oo.plot(raline,deline,transform=oo.get_transform('icrs'))
        oo.set_title(name,fontsize=20)
        ra_s = cc_new.ra.to_string(unit=u.hour,sep=':')
        de_s = cc_new.dec.to_string(unit=u.degree,sep=':')
        #ra_s = [alltargs_uniq.loc[i,'cc'].ra.to_string(unit=u.hour,sep=':') for i in ai]
        #oo.text(0,-.1,'TEST',transform=oo.transAxes)
        oo.text(0,-.1,'RA (ICRS): '+ra_s,transform=oo.transAxes,fontsize=20)
        oo.text(0,-.15,'DEC (ICRS): '+de_s,transform=oo.transAxes,fontsize=20)
        #oo.text(0,-.2,'FOV radius: {}'.format(fovrad.arcsec)
        fig.savefig(outdir+name+'.pdf')
        close(fig)
        return cc_new
    
    def make_chart_shift(self,ra,de,pmra,pmde,name,outdir = 'findercharts_2/'):
        # make the chart with the 
        ra_ang = Angle(degrees=ra)
        de_ang = Angle(degrees=de)
        star_obj = Star(ra=ra_ang,dec=de_ang,ra_mas_per_year=pmra,dec_mas_per_year=pmde)
        astrometric = self.earth.at(self.t).observe(star_obj)
        astrometric_1989 = self.earth.at(self.t_1989).observe(star_obj)
        ra_out,dec_out,dist_out = astrometric.radec()
        ra_out1,dec_out1,dist_out1 = astrometric_1989.radec()
        ra_new = ra_out.to(u.degree).value
        de_new = dec_out.to(u.degree).value
        ra_old = ra_out1.to(u.degree).value
        de_old = dec_out1.to(u.degree).value
        raline = [ra_new,ra_old]#[ra_old,ra_new]
        deline = [de_new,de_old]#[de_old,de_new]
        cc = astropy.coordinates.SkyCoord(ra,de,unit=(u.degree,u.degree))
        cc_new = astropy.coordinates.SkyCoord(ra_new,de_new,unit=(u.degree,u.degree))
        cc_old = astropy.coordinates.SkyCoord(ra_old,de_old,unit=(u.degree,u.degree))
        fig=figure(figsize=(10,12))
        fovrad = astropy.coordinates.Angle('0d4m0s')
        if isfinite(pmra):
            #print 'pm finite'
            if max(abs(pmra),abs(pmde)) > 1000:
                fovrad = astropy.coordinates.Angle('0d8m0s')
            if max(abs(pmra),abs(pmde)) > 4000:
                print 'big radius'
                fovrad = astropy.coordinates.Angle('0d16m0s')
        oo,oh = astroplan.plots.plot_finder_image(cc_new,reticle=True,survey='DSS2 Red',fov_radius=fovrad)
        oo.set_autoscale_on(False)
        oo.plot(raline,deline,transform=oo.get_transform('icrs'))
        oo.set_title(name,fontsize=20)
        ra_s = cc_new.ra.to_string(unit=u.hour,sep=':')
        de_s = cc_new.dec.to_string(unit=u.degree,sep=':')
        #ra_s = [alltargs_uniq.loc[i,'cc'].ra.to_string(unit=u.hour,sep=':') for i in ai]
        #oo.text(0,-.1,'TEST',transform=oo.transAxes)
        oo.text(0,-.1,'RA (ICRS): '+ra_s,transform=oo.transAxes,fontsize=20)
        oo.text(0,-.15,'DEC (ICRS): '+de_s,transform=oo.transAxes,fontsize=20)
        #oo.text(0,-.2,'FOV radius: {}'.format(fovrad.arcsec)
        fig.savefig(outdir+name+'.pdf')
        close(fig)
        return cc_new
        
    def make_chart_sdss(self,ra,de,pmra,pmde,name,outdir = 'findercharts_1d/'):
        ra_ang = Angle(degrees=ra)
        de_ang = Angle(degrees=de)
        star_obj = Star(ra=ra_ang,dec=de_ang,ra_mas_per_year=pmra,dec_mas_per_year=pmde)
        astrometric = self.earth.at(self.t).observe(star_obj)
        astrometric_old = self.earth.at(self.t_2000).observe(star_obj)
        ra_out,dec_out,dist_out = astrometric.radec()
        ra_out1,dec_out1,dist_out1 = astrometric_old.radec()
        ra_new = ra_out.to(u.degree).value
        de_new = dec_out.to(u.degree).value
        ra_old = ra_out1.to(u.degree).value
        de_old = dec_out1.to(u.degree).value
        raline = [ra_old,ra_new]
        deline = [de_old,de_new]
        cc = astropy.coordinates.SkyCoord(ra,de,unit=(u.degree,u.degree))
        cc_new = astropy.coordinates.SkyCoord(ra_new,de_new,unit=(u.degree,u.degree))
        cc_old = astropy.coordinates.SkyCoord(ra_old,de_old,unit=(u.degree,u.degree))
        fig=figure(figsize=(10,12))
        fovrad = astropy.coordinates.Angle('0d4m0s')
        oo,oh = astroplan.plots.plot_finder_image(cc_old,reticle=True,survey='2MASS-K')
        oo.set_autoscale_on(False)
        oo.plot(raline,deline,transform=oo.get_transform('icrs'))
        oo.set_title(name,fontsize=20)
        ra_s = cc_new.ra.to_string(unit=u.hour,sep=':')
        de_s = cc_new.dec.to_string(unit=u.degree,sep=':')
        #ra_s = [alltargs_uniq.loc[i,'cc'].ra.to_string(unit=u.hour,sep=':') for i in ai]
        #oo.text(0,-.1,'TEST',transform=oo.transAxes)
        oo.text(0,-.1,'RA (ICRS): '+ra_s,transform=oo.transAxes,fontsize=20)
        oo.text(0,-.15,'DEC (ICRS): '+de_s,transform=oo.transAxes,fontsize=20)
        fig.savefig(outdir+name+'.pdf')
        close(fig)

    
def print_phase2_targs(inp,repeat_exposures,output,program='PSU18-1-001',instrument="HPF",visits=1):
    """
    Save TSL file

    INPUT:
    *inp* - a pandas dataframe with the following keywords:
    -- 'main_id'
    -- 'kp'
    -- 'pri'
    -- 'cc_new'
    -- 'exptime'
    - repeat_exposures - an array describing how often to repeat each exposure (assumes LRS-2B and R are repeated as often)

    *output* - a string containing the .tsl filename

    OUTPUT:
    - Saves a .tsl file with the name *output*

    EXAMPLE:
        df_all["main_id"] = ["EPIC_"+str(name)[0:-2] for name in df_all["epic"].values]
        df_all["exptime"] = 150*np.ones(len(df_all))
        df_all["kp"]      = df_all["kp"].values
        df_all["cc_new"]  = [SkyCoord(df_all.get_value(i,"k2_ra"),df_all.get_value(i,"k2_dec"),unit=(u.deg, u.deg)) for i in df_all.index]
        df_all["pri"]     = 4.*np.ones(len(df_all))
        HETobs.het_obs.print_phase2_targs(df_all,"lrs2_submission.tsl")

    NOTES:
    """
    oo = []
    oo.append('COMMON')
    oo.append('   PROGRAM ' + program)
    oo.append('   SEEING 3.0')
    oo.append('   SKYTRANS S')
    oo.append('   NUMEXP 1')
    oo.append('   VISITS '+str(int(visits)))
    oo.append('   STDCALS Y')
    oo.append('   SNWAVE 6000')
    oo.append('   SNGOAL 200')
    #oo.append('   PRI 2')
    oo.append('   SKYBRIGHT 18')
    oo.append('   SKYCALS Y')
    oo.append('   TELL Y')
    #oo.append('TRACK_LIST')
    #oo.append('   OBJECT RA DEC GNAME GTYPE INSTRUMENT EXP MAG')
    px = '      '
    for i in inp.index:
        if instrument == "LRS2":
            oo.append('GROUP')
            oo.append('   GNAME '+inp.loc[i,'main_id'])
            oo.append('   GTYPE SEQ')
        oo.append('TRACK')
        oo.append('   OBJECT '+inp.loc[i,'main_id'])
        oo.append('   MAG '+'{:.0f}'.format(inp.loc[i,'kp']))
        #oo.append('   EXPTIME '+'{:.0f}'.format(inp.loc[i,'exptime']))
        oo.append('   PRI '+'{:.0f}'.format(inp.loc[i,'pri']))
        oo.append('   RA '+inp.loc[i,'cc_new'].ra.to_string(unit=u.hour,sep=':',precision=2))
        oo.append('   DEC '+inp.loc[i,'cc_new'].dec.to_string(unit=u.degree,sep=':',precision=2))
        oo.append('ACTION_LIST')
        oo.append('   INSTRUMENT      EXP')
        if instrument=="HPF":
            for j in range(int(repeat_exposures[i])):
                oo.append('      HPF '+'{:.0f}'.format(inp.loc[i,'exptime']))
        elif instrument=="LRS2":
            for j in range(int(repeat_exposures[i])):
                oo.append('      LRS2-R '+'{:.0f}'.format(inp.loc[i,'exptime']))
            for j in range(int(repeat_exposures[i])):
                oo.append('      LRS2-B '+'{:.0f}'.format(inp.loc[i,'exptime']))
        else:
            print("Instrument must be either 'HPF' or 'LRS2'")
            raise Exception
    oo_all = '\n'.join(oo)
    f = open(output,'w')
    f.write(oo_all)
    f.close()
    
    
    
