from __future__ import print_function
import pandas as pd
import numpy as np
import datetime
import astropy.units as u
from astropy.time import TimeDelta
from astroplan import Observer
import random
import astropy
import matplotlib.pyplot as plt

SIDEREAL_DAY_SECONDS = 86164.1
SOLAR_DAY_SECONDS = 86400.

def plot_het_tracks(df,
                    night_observing="2018-03-22",
                    plot_current_time=True,
                    num_live_update_sec=8*3600.,
                    ref_calc_time="2018-02-27T05:00:00"):
    """
    A function to plot HET track plots, showing the Area of the telescope filled for a given target 
    as a funciton of time. Tracks are calculated once for a reference, and then shifted in sidereal time (~4min night-to-night)
    to get tracks for other nights.
    
    INPUT:
        df: pandas pickled dataframe containing the HET tracks (convenience for now, as the
            tracks are calculation intensive)
                Main columns are:
                    - target: name of target
                    - e_center: East track center
                    - e_xtrack: times in second from center of East track
                    - e_ytrack: area (m^2) of HET of the target at e_xtrack times
                    - e_moon_sep: moon separation of the target at the center of the track (peak)
                    - w_center: West track center
                    - w_xtrack: times in second from center of West track
                    - w_ytrack: area (m^2) of HET of the target at w_xtrack times
                    - w_moon_sep: moon separation of the target at the center of the track (peak)
        night_observing: night of observation in UT time (i.e., night of March 21 2018
                        local-time McDonald is '2018-03-22' UT
        plot_current_time: if True, plot a live update blue line showing time now
        num_live_update_sec: number of seconds to live update plot (default 8 hours)
        ref_calc_time: this refers to the night the tracks were calculated.
        
    OUTPUT:
        Live updating plot if plot_current_time=True. Make sure to use interactive backend (nbagg, osx, GTKAgg etc.)
        
    NOTES:
        Make sure to plot in an interactive backend to be able to zoom in (otherwise labels will really overlap).
        In mac do:
            %matplotlib osx
        In linux I normally just do:
            %matplotlib
        and it defaults to nbagg. Can also try some of the following:
            %matplotlib nbagg
            plt.get_backend()
        you can find all backends by doing:
        matplotlib.rcsetup.all_backends
    """
    # Define color scheme and colors to use for each target (West and East tracks are colored same for each target)
    cmap = plt.get_cmap("Dark2")
    number = len(df)
    colors = [cmap(i) for i in np.random.uniform(low=0.0,high=1.0,size=number)]
    df["color"] = colors
    df_e = df[["target","e_center","e_xtrack","e_ytrack","e_moon_sep","color"]]
    df_e = df_e.dropna()
    df_w = df[["target","w_center","w_xtrack","w_ytrack","w_moon_sep","color"]]
    df_w = df_w.dropna()
    
    # Add 5 hours so that it is during the middle of the night at HET. Works better for astroplan algorithms
    time_observing = night_observing + " 05:00:00" 
    
    # Get number of days from reference date; 
    # we offset the tracks the equivalent number of sidereal days from that date
    T_ref = astropy.time.Time(ref_calc_time,scale="utc")
    T_obs = astropy.time.Time(time_observing,scale="utc")
    delta_days = int((T_obs-T_ref).value)
    print("Delta days: {}".format(delta_days))

    # Calculate sunrise and sunset times
    McD = Observer.at_site('McDonald Observatory')
    sun_set_time_00 = McD.sun_set_time(T_obs,which='nearest',horizon=-0*u.deg)
    sun_set_time_06 = McD.sun_set_time(T_obs,which='nearest',horizon=-6*u.deg)
    sun_set_time_12 = McD.sun_set_time(T_obs,which='nearest',horizon=-12*u.deg)
    sun_set_time_18 = McD.sun_set_time(T_obs,which='nearest',horizon=-18*u.deg)
    sun_rise_time_00 = McD.sun_rise_time(T_obs,which='nearest',horizon=-0*u.deg)
    sun_rise_time_06 = McD.sun_rise_time(T_obs,which='nearest',horizon=-6*u.deg)
    sun_rise_time_12 = McD.sun_rise_time(T_obs,which='nearest',horizon=-12*u.deg)
    sun_rise_time_18 = McD.sun_rise_time(T_obs,which='nearest',horizon=-18*u.deg)
    print("Sunset",sun_set_time_00.iso)
    print("Sunrise",sun_rise_time_00.iso)
    
    # Plot East tracks
    for i, name in enumerate(df_e.target):
        # Offset times: dotted lines
        times = df_e["e_center"].values[i] + TimeDelta(df_e["e_xtrack"].values[i]*u.second) + \
                TimeDelta(delta_days*SIDEREAL_DAY_SECONDS*u.second)
        plt.plot(times.datetime,df_e["e_ytrack"].values[i],color=df_e.color.values[i],ls="--")
        title = name + " ({0:0.2f})".format(df_e.e_moon_sep.values[i])
        ii = random.randint(50,55) # pick random index so that the labels don't all overlap
        plt.text(times.datetime[ii],df_e["e_ytrack"].values[i][ii],title,fontsize=8,color=df_e.color.values[i])
        ii = random.randint(0,10)  # pick random index so that the labels don't all overlap
        plt.text(times.datetime[ii],df_e["e_ytrack"].values[i][ii],title,fontsize=8,color=df_e.color.values[i])

    # Plot West tracks: solid lines
    for i, name in enumerate(df_w.target):
        times = df_w["w_center"].values[i] + TimeDelta(df_w["w_xtrack"].values[i]*u.second) + \
                     TimeDelta(delta_days*SIDEREAL_DAY_SECONDS*u.second)
        plt.plot(times.datetime,df_w["w_ytrack"].values[i],color=df_w.color.values[i])
        title = name + " ({0:0.2f})".format(df_w.w_moon_sep.values[i])
        ii = random.randint(50,55) # pick random index so that the labels don't all overlap
        plt.text(times.datetime[ii],df_w["w_ytrack"].values[i][ii],title,fontsize=8,color=df_w.color.values[i])
        ii = random.randint(0,10)  # pick random index so that the labels don't all overlap
        plt.text(times.datetime[ii],df_w["w_ytrack"].values[i][ii],title,fontsize=8,color=df_w.color.values[i])

    ylim = (20,52)
    plt.ylim(ylim[0],ylim[1])
    plt.fill_betweenx(np.linspace(ylim[0],ylim[1],2),sun_set_time_00.datetime,sun_set_time_06.datetime,alpha=0.05,color="k")
    plt.fill_betweenx(np.linspace(ylim[0],ylim[1],2),sun_set_time_06.datetime,sun_set_time_12.datetime,alpha=0.1,color="k")
    plt.fill_betweenx(np.linspace(ylim[0],ylim[1],2),sun_set_time_12.datetime,sun_set_time_18.datetime,alpha=0.15,color="k")
    plt.fill_betweenx(np.linspace(ylim[0],ylim[1],2),sun_set_time_18.datetime,sun_rise_time_18.datetime,alpha=0.2,color="k")
    plt.fill_betweenx(np.linspace(ylim[0],ylim[1],2),sun_rise_time_18.datetime,sun_rise_time_12.datetime,alpha=0.15,color="k")
    plt.fill_betweenx(np.linspace(ylim[0],ylim[1],2),sun_rise_time_12.datetime,sun_rise_time_06.datetime,alpha=0.1,color="k")
    plt.fill_betweenx(np.linspace(ylim[0],ylim[1],2),sun_rise_time_06.datetime,sun_rise_time_00.datetime,alpha=0.05,color="k")
    plt.grid(lw=0.5,alpha=0.9)
    plt.xlabel("Time [UT]")
    plt.ylabel("Aperture (m^2)")
    plt.minorticks_on()
    plt.show(block=False)
    
    title = "HET Track plot for {}\n".format(time_observing[0:10])
    title += "Sunset (-6deg): {}\n".format(sun_set_time_06.iso)
    title += "Sunrise (-6deg): {}\n".format(sun_rise_time_06.iso)
    
    # 8 hours
    if plot_current_time:
        for i in range(num_live_update_sec/2):
            time_now = datetime.datetime.utcnow()
            time_now = astropy.time.Time(time_now)
            time_now = time_now.datetime
            timeline = plt.axvline(time_now,color="blue")
            _title=title+"Last updated: "+str(time_now)
            plt.title(_title)
            plt.draw()
            plt.pause(2)
            timeline.remove()
            plt.tight_layout()
    else:
        plt.title(title)
        plt.tight_layout()
