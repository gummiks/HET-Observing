# HET-Observing
Useful scripts for observing with The Hobby-Eberly Telescope.

Currently includes the following capabilities:
- Create TSL file from a list of stars in a .csv file
- Query simbad resolvable stars to look for magnitudes, ra, dec, pmra, pmdec
- Check observability for a given set of dates
- Create findercharts
- Account for proper motions at a given epoch (HET needs current epoch RA/DEC)

# Example HET TSL Files and FinderCharts
See notebooks on creating an LRS2 TSL files / findercharts.
Additionally, a notebook on creating HPF engineering-run TSL files is included (some caveats as HPF is currently being commissioned; see notebook).

# Getting started
Take a look at the ipynb notebooks in `/notebooks`

# Dependencies
- astropy: http://www.astropy.org/
- astroplan: https://astroplan.readthedocs.io/en/latest/
- astroquery: http://astroquery.readthedocs.io/en/latest/
- skyfield: http://rhodesmill.org/skyfield/
- pandas
- tqdm

### Todo
- Add Joe Ninan's pyHETObs code for track filling

# Coauthors
- Ryan Terrien
- Joe Ninan
