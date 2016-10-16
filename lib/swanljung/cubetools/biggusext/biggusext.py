"""Extensions to biggus."""
from __future__ import absolute_import, division, print_function
from six.moves import (filter, input, map, range, zip)
import six

import biggus
import numpy as np
import numpy.ma as ma

biggus._init.MAX_CHUNK_SIZE *= 10

class IndexGroupedArray(biggus.Array):
    """ This class should have a better name. """

    def __init__(self, source, slices, groupfun=biggus.mean, axis=0):
        self.source = source
        self.slices = np.array(slices)
        self.axis = axis
        self.groupfun = groupfun

    @property
    def dtype(self):
        """The datatype of this virtual array."""
        return self.source.dtype

    @property
    def shape(self):
        """The shape of the virtual array as a tuple."""
        ss = list(self.source.shape)
        ss[self.axis] = len(self.slices)
        return tuple(ss)

    def _getitem_full_keys(self, keys):
        """
        Returns a new Array by slicing this virtual array.

        Parameters
        ----------
        keys - iterable of keys
            The keys to index the array with. The default ``__getitem__``
            removes all ``np.newaxis`` objects, and will be of length
            array.ndim.

        Note: This method must be overridden if ``__getitem__`` is defined by
        :meth:`Array.__getitem__`.

        """
        def ssta(aslice, axis):
            if isinstance(aslice, slice):
                start = aslice.start if aslice.start else 0
                stop = aslice.stop if aslice.stop else self.source.shape[axis]
                step = aslice.step if aslice.step else 1
                return np.arange(start, stop, step)
            else:
                return aslice

        if isinstance(keys[self.axis], int):
            okeys = [ssta(k, i) for i, k in enumerate(keys)]
            okeys[self.axis] = self.slices[keys[self.axis]]
            return self.groupfun(self.source[okeys], self.axis)
        else:
            okeys = [ssta(k, i) for i, k in enumerate(keys)]
            okeys[self.axis] = np.arange(self.source.shape[self.axis])
            return IndexGroupedArray(self.source[tuple(okeys)],
                                     self.slices[keys[self.axis]],
                                     self.groupfun, self.axis)

    def ndarray(self):
        """
        Returns the NumPy ndarray instance that corresponds to this
        virtual array.

        """
        return self._getarray()

    def _getarray(self, masked=False):
        a = np.zeros(self.shape, dtype=self.dtype)
        if masked:
            a = ma.MaskedArray(a)
        aslice = [slice(None) for d in self.shape]
        sslice = [np.arange(d) for d in self.source.shape]
        for i, s in enumerate(self.slices):
            if isinstance(s, slice):
                sslice[self.axis] = np.arange(s.start, s.stop, s.step)
            else:
                sslice[self.axis] = np.array(s[0])
            aslice[self.axis] = i
            ss = self.source[tuple(sslice)]
            bg = self.groupfun(ss, self.axis)
            if masked:
                q = bg.masked_array()
                a[tuple(aslice)] = q
            else:
                a[tuple(aslice)] = bg.ndarray()
        return a

    def masked_array(self):
        """
        Returns the NumPy MaskedArray instance that corresponds to this
        virtual array.

        """
        return self._getarray(masked=True)