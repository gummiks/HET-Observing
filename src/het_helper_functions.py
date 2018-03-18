from __future__ import print_function
import astroquery
import pandas as pd
from astroquery.simbad import Simbad
import numpy as np
import het_obs
import astropy.coordinates
import astropy.units as u
import astropy.coordinates
import het_config
from skyfield.api import load, Star, Angle
import astropy.units as u
import astropy
MAX_ALT = het_config.HET_MAX_ALT
MIN_ALT = het_config.HET_MIN_ALT

compactString = lambda string: string.replace(' ', '_')

def raHMS2deg(angle):
    """
    EXAMPLE: Angle("10:42:44",unit=u.hourangle).deg
    """
    return astropy.coordinates.Angle(angle,unit=u.hourangle).deg

def decDeg2deg(angle):
    """
    EXAMPLE: Angle("10:42:44",unit=u.hourangle).deg
    """
    return astropy.coordinates.Angle(angle,unit=u.deg).deg

def radec2degrees(ra,dec):
    """
    Convert Hourangle RA and DMS DEC to floating point degrees

    INPUT:
    - ra (str) in Hourangles, e.g., '04 13 05.604'
    - dec (str) in DMS, e.g., +15 14 52.02'

    OUTPUT:
    - ra in degrees
    - dec in degrees

    EXAMPLE:
    radec2deg('04 13 05.604','+15 14 52.02')
    (63.27334999999999, 15.247783333333333)
    """
    c = astropy.coordinates.SkyCoord(ra,dec,unit=(u.hourangle,u.degree))
    return c.ra.deg, c.dec.deg

def get_simbad_fluxes(name):
    """
    A function to query simbad for fluxes in magnitudes.
    Notes: function can return more than one star (e.g., if binary)

    INPUT:
        Simbad resolvable name of the star. 

    OUTPUT:
        a pandas dataframe with the flux values

    """
    customSimbad = Simbad()
    customSimbad.add_votable_fields('flux(V)','flux(R)','flux(I)','flux(J)','flux(H)','flux(g)','flux(r)','flux(i)')
    table = customSimbad.query_object(name)
    columns = ["FLUX_V","FLUX_R","FLUX_I","FLUX_J","FLUX_H","FLUX_g","FLUX_r","FLUX_i"]
    df = table[columns].to_pandas()
    return df

def get_ra_dec_pmra_pmdec(name,epoch=2018.07):
    """
    Get RA, DEC, PMRA and PMDEC from SIMBAD using astroquery

    INPUT:
    - name of the star
    - epoch - the epoch of the observation, assumes J2018.07 (Jan 2018), and assumes equinox = 2000

    OUTPUT:
    - astropy table with the columns:
    -- MAIN_ID
    -- RA
    -- DEC
    -- PMRA (mas/yr)
    -- PMDEC (mas/yr)
    -- RA_2_A_ICRS_J2018_07_2000
    -- DEC_2_D_ICRS_J2018_07_2000
    """
    customSimbad = Simbad()
    ra_epoch_string  = "ra(2;A;ICRS;J"+str(epoch)+";2000)"
    dec_epoch_string = "dec(2;D;ICRS;J"+str(epoch)+";2000)"
    fields = ("pmra","pmdec",ra_epoch_string,dec_epoch_string)
    customSimbad.add_votable_fields(*fields)
    table = customSimbad.query_object(name)
    #table[["RA","DEC","RA_2_A_ICRS_J2018_07_2000","DEC_2_D_ICRS_2018_07_2000","PMRA","PMDEC"]]
    return table

def filter_het_observable(df,tr_midp_sun_alt=-18.):
    return df[(df["tr_midp_alt"]>MIN_ALT) & (df["tr_midp_alt"]<MAX_ALT) & (df["tr_midp_sun_alt"] < tr_midp_sun_alt)]

class AngleWithProperMotion(object):
    """
    A class to work with RA/DEC with proper motions

    EXAMPLE:
        df_e = astropylib.k2help.get_epic_info([201885041])
        A = AngleWithProperMotion()
        cnew = A.radec_with_proper_motion(df_e.k2_ra.values[0],
                                   df_e.k2_dec.values[0],
                                   df_e.pmra.values[0],
                                   df_e.pmdec.values[0])
    """

    def __init__(self,epoch=2018.):
        planets = load('de421.bsp')
        ts = load.timescale()
        self.earth = planets['earth']
        self.epoch_new = epoch
        self.epoch_old = 2000.
        self.t = ts.utc(self.epoch_new)
        # these two epochs work for most of the finder charts
        self.t_1989 = ts.utc(1990) # was 1989, but stars were offset bc of varying epoch
        self.t_2000 = ts.utc(self.epoch_old)
        print("Loaded de421.bsp")

    def radec_with_proper_motion(self,ra,de,pmra,pmde,verbose=False):
        """
        Calculate the

        INPUT:
        - ra in degrees
        - dec in degrees
        - pmra - in mas/yr
        - pmdec - in mas/yr

        OUTPUT:
        - Angle

        NOTES:
        - Built on Ryan's code
        """
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
        self.cc_old = cc_old
        self.cc_new = cc_new
        self.ra_s_new = cc_new.ra.to_string(unit=u.hour,sep=':')
        self.de_s_new = cc_new.dec.to_string(unit=u.degree,sep=':')
        self.ra_s_old = cc_old.ra.to_string(unit=u.hour,sep=':')
        self.de_s_old = cc_old.dec.to_string(unit=u.degree,sep=':')
        
        if verbose:
            print("Coordinates at {:f}: RA={} DEC={}".format(self.epoch_old,self.ra_s_old,self.de_s_old))
            print("with  PMRA={} PMDEC={}".format(pmra,pmde))
            print("Coordinates at {:f}: RA={} DEC={}".format(self.epoch_new,self.ra_s_new,self.de_s_new))
            print("")
        return cc_new

def lrs2_scale_exptime_from_mags(m2,m1=14,t1=200.,MINLIM=50.,MAXLIM=500.,useceil=True,verbose=True):
    """
    Scale exposure times for a given reference exposure time for different magnitudes

    INPUT:
        m2 - the magnitude of the star you are scaling exposure time to
        m1 - mag of reference star
        t1 - exposure time for reference star
        MINLIM - min exposure time
        MAXLIM - max exposure time
        useceil - round off values if true
        verbose - print useful statements

    NOTES:
        Useful for LRS2 submission

    EXAMPLE:
        scale_exptime_from_mags(13.,14.,200.)
    """
    t2 = t1*(10.**((m2-m1)/2.5))
    if useceil: t2 = np.ceil(t2)
    if t2 > MAXLIM:
        if verbose: print("Warning: t2 =",t2,"setting to =",MAXLIM)
        t2=MAXLIM
    if t2 < MINLIM:
        if verbose: print("Warning: t2 =",t2,"setting to =",MINLIM)
        t2=MINLIM
    return t2

def print_TSL(names,vmag,priority,ra,dec,pmra,pmdec,exptime,repeat_exposures,outputname,epoch=2018.07,program="PSU18-1-001",instrument="HPF",visits=1):
    """
    A helper function to print TSL files

    INPUT:
        names      - name of the target without spaces
        vmag       - an array of vmags
        priority   - an array of priorities: can be 1,2,3, or 4
        ra         - ra in degrees, J2000
        dec        - dec in degrees, J2000
        pmra       - in mas/yr
        pmdec      - in mas/yr
        exptime    - exposure time in s
        outputname - filename to save
        epoch      - epoch of observation (=2018.07 for Jan 26 2018)

    OUTPUT:
        saves the TSL file to output

    EXAMPLE:
        # Epoch for January 26, 2018
        epoch = 2018. + 26./365.245
        names   = df["MAIN_ID"].values
        vmag    = df["VMAG"].values
        priority= 4*np.ones(len(df))
        ra      = df["RA_SIMBAD"].values
        dec     = df["DEC_SIMBAD"].values
        pmra    = df["PMRA"].values
        pmdec   = df["PMDEC"].values
        exptime = 100.*np.ones(len(df))
        print_TSL(names,vmag,priority,ra,dec,pmra,pmdec,exptime,outputname="../data/lrs2/tsl/lrs2_submission_kdwarfs.tsl",epoch=epoch)
    """
    print("Number of visits = {}".format(visits))
    print("Instrument = {}".format(instrument))
    print("Program = {}".format(program))
    A = AngleWithProperMotion(epoch=epoch)
    c = [A.radec_with_proper_motion(ra[i],dec[i],pmra[i],pmdec[i]) for i in range(len(ra))]
    df = pd.DataFrame(zip(names,vmag,priority,c,exptime),columns=["main_id","kp","pri","cc_new","exptime"])
    het_obs.print_phase2_targs(df,repeat_exposures,outputname,program=program,instrument=instrument,visits=visits)
    print("Printed TSL file to {}".format(outputname))


def arange_time(t_start,t_stop,delta=1.,form="datetime"):
    """
    Get a list of times starting at t_start, ending at t_top, with a delta of *delta* in days.
    Format can be astropy.time attributes

    INPUT:
        start: '2018-02-25 05:00:00'
        stop: '2018-03-06 05:00:00'
        delta: day

    OUTPUT:
        times: range of times of a given format

    EXAMPLEs:
        arange_time('2018-02-25 05:00:00','2018-03-05 05:00:00',form="iso")
        arange_time('2018-02-25 05:00:00','2018-03-05 05:00:00',form="jd")
        arange_time('2018-02-25 05:00:00','2018-03-05 05:00:00',form="datetime")
        arange_time('2018-02-25 05:00:00','2018-03-05 05:00:00')
    """
    t_start = astropy.time.Time(t_start,format="iso",scale="utc").jd
    t_stop  = astropy.time.Time(t_stop,format="iso",scale="utc").jd
    times_jd = np.arange(t_start,t_stop,delta)
    if form is None:
        times = [astropy.time.Time(time,format="jd",scale="utc") for time in times_jd]
    else:
        times = [getattr(astropy.time.Time(time,format="jd",scale="utc"),form) for time in times_jd]
    return np.array(times)

