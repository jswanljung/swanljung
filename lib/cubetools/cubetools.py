""" Functions for performing common tasks with cubes, in a hopefully fast and
    memory-efficient way.
"""

# daily climatology
# map function biggus array subclass
# daily_max
# daily_mean
# daily_min
# climatology of any of those
# standard deviation of any of those
from __future__ import absolute_import, division, print_function
from six.moves import (filter, input, map, range, zip)
import six

import iris
import iris.experimental.equalise_cubes as ieec
from datetime import datetime

import numpy as np
import numpy.ma as ma
import sys
from scipy.signal import butter, filtfilt
import biggus

from .biggusext import IndexGroupedArray

iris.FUTURE.cell_datetime_objects = True


def timestamp():
    return datetime.utcnow().strftime('%Y-%m%d %H:%M:%S UTC')


def runlogline():
    """ Return a line with timestamp and concatenated sys.argv """
    rll = [timestamp]
    rll.extend(sys.argv)
    return " ".join(rll)


def append_rll(cube):
    h = runlogline()
    if cube.attributes['history']:
        h = cube.attributes['history'] + "\n" + h
    cube.attributes['history'] = h


def add_misu(cube):
    cube.attributes['institution'] = ("Stockholm University, Department "
                                      "of Meteorology (MISU)")


# Some of these utilities should probably be put in another module
def day_of_year(d, leapday=366):
    """ Calculate ordinal day_of_year from a datetime.

        d: a datetime.datetime or datetime.date object

        leapday: The ordinal day of year that should be considered
            the leap day and assigned the number 366. When we do
            climatologies we're going to drop leap days, but we
            don't want the dropped days to occur in the middle of
            a period of interest. By default this is the last day
            of the year, but by specifying another number we can
            put the leap day anywhere we want it.

        returns: an integer day of year (with 366 assigned to the
            leap day.)
    """
    firstday = datetime(d.year, 1, 1).toordinal()
    doy = d.toordinal() - firstday + 1
    if doy > leapday:
        return doy - 1
    elif doy == leapday:
        return 366
    return doy


def day_of_year_to_monthday(doy, leap=False):
    """Return a (month,day) tuple."""
    year = 2001 if not leap else 2000
    dayo = datetime(year, 1, 1).toordinal() + doy - 1
    day = datetime.fromordinal(dayo)
    return (day.month, day.day)


def crazy_concatenate(cubelist, axis=0, cubesortkey=None):
    """ Concatenate cubes along an axis without any checks."""
    if not cubesortkey:

        def cubesortkey(cube):
            return cube.dim_coords[axis].points[0]

    cubelist.sort(key=cubesortkey)
    refcube = cubelist[-1]
    coords = refcube.dim_coords
    allcpoints = [cube.dim_coords[axis].points for cube in cubelist]
    points = np.concatenate(allcpoints, axis)
    coords[axis] = coords[axis].copy(points)
    alldata = [cube.lazy_data() for cube in cubelist]
    data = biggus.LinearMosaic(alldata, axis)
    crazycube = iris.Cube(data, standard_name=refcube.standard_name,
                          long_name=refcube.long_name,
                          var_name=refcube.var_name,
                          units=refcube.units,
                          dim_coords_and_dims=[(c, i) for i, c in
                                               enumerate(coords)])
    return crazycube



def super_equalise(cubelist):
    """ Equalises cube attributes using the iris function,
        but also makes sure all names and units match, standardizing
        to the values in the first cube. No sanity checking of any kind
        is performed, so only do this if you know it makes sense. The
        modification is in-place.

        cubelist: An iterable of cubes.
    """
    refcube = cubelist[-1]
    standard_name = refcube.standard_name
    long_name = refcube.long_name
    var_name = refcube.var_name
    units = refcube.units

    for cube in cubelist:
        cube.standard_name = standard_name
        cube.long_name = long_name
        cube.var_name = var_name
        cube.units = units

    ieec.equalise_attributes(cubelist)
    _equalise_coords(cubelist)


def _equalise_coords(cubelist):
    refcube = cubelist[-1]
    rcs = refcube.coords()

    for cube in cubelist:
        for co, rc in zip(cube.coords(), rcs):
            if cube.coord_dims(co.name()) == refcube.coord_dims(rc.name()):
                co.standard_name = rc.standard_name
                co.long_name = rc.long_name
                co.var_name = rc.var_name
                co.units = rc.units
            co.attributes.clear()
        cube.coord('time').points = cube.coord('time').points.astype('int32')


def _coorddatetimes(timecoord):
    return timecoord.units.num2date(timecoord.points)


def _slicesfromdates(dates):
    ordinals = np.array([d.toordinal() for d in dates])
    du, di = np.unique(ordinals, return_index=True)
    di = np.concatenate((di, [len(ordinals)]))
    slices = [slice(di[i], di[i+1]) for i in range(len(di)-1)]
    return slices


def _dayslices(timecoord):
    """ Return a list of slices corresponding to days. """
    dates = _coorddatetimes(timecoord)
    return _slicesfromdates(dates)


def _monthslices(timecoord):
    """ Return a list of slices corresponding to months. """
    dates = _coorddatetimes(timecoord)
    monthdays = [datetime(d.year, d.month, 1) for d in dates]
    return _slicesfromdates(monthdays)


def _yearslices(timecoord):
    dates = _coorddatetimes(timecoord)
    yeardays = [datetime(d.year, 1, 1) for d in dates]
    return _slicesfromdates(yeardays)


def _seasonslices(timecoord,
                  seasons=('djf', 'mam', 'jja', 'son')):
    sa = np.arange(1, 13)
    offset = ('jfmamjjasond'*2).index("".join(seasons))
    sa = np.roll(sa, -offset)
    i = 0
    for j, s in enumerate(seasons):
        L = len(s)
        sa[i:i+len(s)] = sa[i]
        i += L
    sa = np.roll(sa, offset)

    def smonth(month):
        return sa[month-1]

    def sdate(dd):
        m = smonth(dd.month)
        y = dd.year if m <= dd.month else dd.year-1
        return datetime(y, m, 1)

    dates = _coorddatetimes(timecoord)
    sdays = [sdate(d) for d in dates]
    return _slicesfromdates(sdays)


def _newtimecoord(timecoord, slices, setbounds=False, uselast=False):
    p = timecoord.points
    if uselast:
        newpoints = np.array([np.max(p[s]) for s in slices])
    else:
        newpoints = np.array([np.median(p[s]) for s in slices])
    boundarray = None
    if setbounds:
        bounds = [p[s.start] for s in slices]
        bounds.append(bounds[-1] + (bounds[1]-bounds[0]))
        boundarray = np.zeros((len(newpoints), 2))
        boundarray[:, 0] = bounds[:-1]
        boundarray[:, 1] = bounds[1:]
    newtime = timecoord.copy(newpoints, boundarray)
    return newtime


def _datecategorytuples(timecoord, catfun):
    """ Generate a list of tuples of indices sorted by category.
        timecoord: an iris time coordinate
        catfun: a function that accepts a datetime and
            returns an integer representing a category.
    """
    dates = _coorddatetimes(timecoord)
    cats = np.array([catfun(d) for d in dates])
    catset = np.unique(cats)
    return [np.where(cats == cs) for cs in catset]


def _doytuples(timecoord, leapday=366):
    """ Generate a list of tuples of day of year indices.

        timecoord: an Iris time coordinate.

        leapday: see docs for day_of_year.
        returns: a list of tuples of indices for each day of year
    """
    dct = _datecategorytuples(timecoord, lambda d: day_of_year(d, leapday))
    return dct[0:365]


def _monthtuples(timecoord):
    """ Generate a list of tuples by month.
    """
    return _datecategorytuples(timecoord, lambda d: d.month)


def _seasontuples(timecoord,
                  seasons=('djf', 'mam', 'jja', 'son')):
    sa = np.zeros(12)
    offset = ('jfmamjjasond'*2).index("".join(seasons))
    i = 0
    for j, s in enumerate(seasons):
        L = len(s)
        sa[i:i+len(s)] = j
        i += L
    sa = np.roll(sa, offset)

    def seasoncat(d):
        return sa[d.month-1]

    return _datecategorytuples(timecoord, seasoncat)


def _chunkcattuples(timecoord, chunklen, leapday=366, offset=0):

    def chunkcat(d):
        doy = day_of_year(d, leapday)
        return (doy+offset) // chunklen

    return _datecategorytuples(timecoord, chunkcat)


def _annualtuples(timecoord):
    return _datecategorytuples(timecoord, lambda d: d.year)


def _mmm(cube, slicefun=_dayslices, timedim="time", mmm='mean'):
    """ Return a cube with chunk averaged maxima, minima, or mean.

    cube: an iris.cube with a time dimension timedim
    timedim: string name of time dimension
    mmm: string, one of 'max', 'min', 'mean'
    """
    funcs = {'max': biggus.max, 'min': biggus.min, 'mean': biggus.mean}
    tc = cube.coord(timedim)
    dayslices = slicefun(tc)
    timeaxis = cube.coord_dims(timedim)[0]
    data = IndexGroupedArray(cube.lazy_data(), dayslices, funcs[mmm],
                             axis=timeaxis)
    newtime = _newtimecoord(tc, dayslices)
    dimcoords = list(cube.dim_coords)
    dimcoords[timeaxis] = newtime
    return iris.cube.Cube(data, standard_name=cube.standard_name,
                          long_name=cube.long_name, var_name=cube.var_name,
                          units=cube.units, attributes=cube.attributes,
                          dim_coords_and_dims=[(d, i) for i, d in
                                               enumerate(dimcoords)])


def daily_max(cube, timedim='time'):
    return _mmm(cube, timedim=timedim, mmm='max')


def daily_min(cube, timedim='time'):
    return _mmm(cube, timedim=timedim, mmm='min')


def daily_mean(cube, timedim='time'):
    return _mmm(cube, timedim=timedim, mmm='mean')


def annual_max(cube, timedim='time'):
    return _mmm(cube, slicefun=_yearslices, timedim=timedim, mmm='max')


def annual_min(cube, timedim='time'):
    return _mmm(cube, slicefun=_yearslices, timedim=timedim, mmm='min')


def annual_mean(cube, timedim='time'):
    return _mmm(cube, slicefun=_yearslices, timedim=timedim, mmm='mean')


def monthly_max(cube, timedim='time'):
    return _mmm(cube, slicefun=_monthslices, timedim=timedim, mmm='max')


def monthly_min(cube, timedim='time'):
    return _mmm(cube, slicefun=_monthslices, timedim=timedim, mmm='min')


def monthly_mean(cube, timedim='time'):
    return _mmm(cube, slicefun=_monthslices, timedim=timedim, mmm='mean')


def seasonal_mean(cube, timedim='time', seasons=('djf', 'mam', 'jja', 'son')):
    def slicefun(d):
        return _seasonslices(d, seasons)
    return _mmm(cube, slicefun=slicefun, timedim=timedim, mmm='mean')


def seasonal_max(cube, timedim='time', seasons=('djf', 'mam', 'jja', 'son')):
    def slicefun(d):
        return _seasonslices(d, seasons)
    return _mmm(cube, slicefun=slicefun, timedim=timedim, mmm='max')


def seasonal_min(cube, timedim='time', seasons=('djf', 'mam', 'jja', 'son')):
    def slicefun(d):
        return _seasonslices(d, seasons)
    return _mmm(cube, slicefun=slicefun, timedim=timedim, mmm='min')


def _climtimecoord(timecoord, indices):
    n2d = timecoord.units.num2date
    d2n = timecoord.units.date2num
    newpoints = [timecoord.points[np.max(i)] for i in indices]
    dates = [n2d(n) for n in newpoints]
    for i in range(len(dates)):
        if dates[i].year != dates[0].year:
            dates[i] = datetime(dates[0].year, dates[i].month, dates[i].day,
                                dates[i].hour)
    newpoints = np.array([d2n(d) for d in dates])
    return timecoord.copy(newpoints)


def _climatology(cube, indexfun=_doytuples, timedim="time",
                 climfun=biggus.mean):
    tc = cube.coord(timedim)
    indextuples = indexfun(tc)
    timeaxis = cube.coord_dims(timedim)[0]
    data = IndexGroupedArray(cube.lazy_data(), indextuples, climfun,
                             axis=timeaxis)
    newtime = _climtimecoord(tc, indextuples)
    dimcoords = list(cube.dim_coords)
    dimcoords[timeaxis] = newtime
    return iris.cube.Cube(data, standard_name=cube.standard_name,
                          long_name=cube.long_name, var_name=cube.var_name,
                          units=cube.units, attributes=cube.attributes,
                          dim_coords_and_dims=[(d, i) for i, d in
                                               enumerate(dimcoords)])


def butterworth_smooth(cube, dim, nykvist_f, cutoff):
    """ Perform butterworth smoothing on a daily climatology. """
    axis = cube.coord_dims(dim)[0]
    b, a = butter(4, cutoff/nykvist_f)
    smoothcube = cube.copy(filtfilt(b, a, cube.data, axis=axis,
                           method='gust'))
    return smoothcube


def climsmooth(cube, timedim='time', months=2):
    nykvist_f = 365/2.
    monthly_f = 12./months
    return butterworth_smooth(cube, timedim, nykvist_f, monthly_f)


def daily_climatology(cube, timedim='time', leapday=366, smooth=False,
                      months=2):
    def doyt(cube):
        return _doytuples(cube, leapday)
    dc = _climatology(cube, doyt, timedim)
    if smooth:
        return climsmooth(dc, timedim, months)
    else:
        return dc


def daily_climatology_std(cube, timedim='time', leapday=366, smooth=False,
                          months=2):
    def doyt(cube):
        return _doytuples(cube, leapday)
    dc = _climatology(cube, doyt, timedim, climfun=biggus.std)
    if smooth:
        return climsmooth(dc, timedim, months)
    else:
        return dc


# This code from iris.analysis shows how to turn multiple slices into a list
# of indices
# def _slice_merge(self):
#     """
#     Merge multiple slices into one tuple and collapse items from
#     containing list.

#     """
#     # Iterate over the ordered dictionary in order to reduce
#     # multiple slices into a single tuple and collapse
#     # all items from containing list.
#     for key, groupby_slices in six.iteritems(self._slices_by_key):
#         if len(groupby_slices) > 1:
#             # Compress multiple slices into tuple representation.
#             groupby_indicies = []

#             for groupby_slice in groupby_slices:
#                 groupby_indicies.extend(range(groupby_slice.start,
#                                               groupby_slice.stop))

#             self._slices_by_key[key] = tuple(groupby_indicies)
#         else:
#             # Remove single inner slice from list.
#             self._slices_by_key[key] = groupby_slices[0]

def happySaturday():
    pass
