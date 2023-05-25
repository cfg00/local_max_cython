%%cython

import numpy as np
from scipy import ndimage

def get_local_max(np.ndarray[np.float64_t, ndim=3] im_dif, double th_fit,
                  np.ndarray[np.float64_t, ndim=3] im_raw=None,
                  dict dic_psf=None, int delta=1, int delta_fit=3,
                  bint dbscan=True, bint return_centers=False, np.ndarray[np.float64_t, ndim=1] mins=None,
                  double sigmaZ=1, double sigmaXY=1.5):

    cdef int zmax = im_dif.shape[0]
    cdef int xmax = im_dif.shape[1]
    cdef int ymax = im_dif.shape[2]

    cdef np.ndarray[np.int_t, ndim=1] z = np.where(im_dif > th_fit)[0]
    cdef np.ndarray[np.int_t, ndim=1] x = np.where(im_dif > th_fit)[1]
    cdef np.ndarray[np.int_t, ndim=1] y = np.where(im_dif > th_fit)[2]

    cdef np.ndarray[np.float64_t, ndim=1] in_im = im_dif[z, x, y]

    cdef int n = len(x)
    cdef np.ndarray[np.bool_t, ndim=1] keep = np.ones(n, dtype=bool)

    cdef int d1, d2, d3
    for d1 in range(-delta, delta + 1):
        for d2 in range(-delta, delta + 1):
            for d3 in range(-delta, delta + 1):
                keep &= (in_im >= im_dif[(z + d1) % zmax, (x + d2) % xmax, (y + d3) % ymax])

    z = z[keep]
    x = x[keep]
    y = y[keep]
    h = in_im[keep]

    Xh = np.column_stack((z, x, y, h))

    cdef np.ndarray[np.float64_t, ndim=2] im_centers
    cdef np.ndarray[np.float64_t, ndim=2] Xft
    cdef np.ndarray[np.float64_t, ndim=1] bk
    cdef np.ndarray[np.float64_t, ndim=1] a
    cdef np.ndarray[np.float64_t, ndim=1] habs
    cdef np.ndarray[np.float64_t, ndim=1] hn
    cdef np.ndarray[np.float64_t, ndim=1] zc
    cdef np.ndarray[np.float64_t, ndim=1] xc
    cdef np.ndarray[np.float64_t, ndim=1] yc

    if delta_fit != 0 and Xh.shape[0] > 0:
        z, x, y, h = Xh.T.astype(np.int32)
        zmax, xmax, ymax = im_dif.shape
        im_centers = np.empty((5, 0), dtype=np.float64)
        Xft = np.empty((0, 3), dtype=np.float64)

        for d1 in range(-delta_fit, delta_fit + 1):
            for d2 in range(-delta_fit, delta_fit + 1):
                for d3 in range(-delta_fit, delta_fit + 1):
                    if d1 * d1 + d2 * d2 + d3 * d3 <= delta_fit * delta_fit:
                        im_centers = np.column_stack((im_centers,
                                                      (z + d1) % zmax,
                                                      (x + d2) % xmax,
                                                      (y + d3) % ymax,
                                                      im_dif[(z + d1) % zmax, (x + d2) % xmax, (y + d3) % ymax]))

                        if im_raw is not None:
                            im_centers = np.column_stack((im_centers,
                                                          im_raw[(z + d1) % zmax, (x + d2) % xmax, (y + d3) % ymax]))

                        Xft = np.vstack((Xft, [d1, d2, d3]))

        bk = np.min(im_centers[3], axis=1)
        im_centers[3] -= bk
        a = np.sum(im_centers[3], axis=1)
        habs = np.zeros_like(bk)

        if im_raw is not None:
            habs = im_raw[z % zmax, x % xmax, y % ymax]

        if dic_psf is not None:
            keys = list(dic_psf.keys())
            im0 = dic_psf[keys[0]]
            space = np.sort(np.diff(keys, axis=0).ravel())
            space = space[space != 0][0]
            zi, xi, yi = (z / space).astype(np.int32), (x / space).astype(np.int32), (y / space).astype(np.int32)
            keys_ = np.array(keys)
            sz_ = np.max(keys_ // space, axis=0) + 1
            ind_ = tuple(Xft.T + (np.array(im0.shape)[:, np.newaxis] // 2 - 1))

            im_psf = np.zeros(sz_ + [len(ind_[0])], dtype=np.float64)
            for key in keys_:
                coord = tuple((key / space).astype(np.int32))
                im__ = dic_psf[tuple(key)][ind_]
                im_psf[coord] = (im__ - np.mean(im__)) / np.std(im__)

            im_psf_ = im_psf[zi, xi, yi]
            im_centers__ = im_centers[3:].T.copy()
            im_centers__ = (im_centers__ - np.mean(im_centers__, axis=1)[:, np.newaxis]) / np.std(im_centers__, axis=1)[:, np.newaxis]
            hn = np.mean(im_centers__ * im_psf_, axis=1)

        else:
            sz = delta_fit
            Xft = np.indices([2 * sz + 1] * 3) - sz
            Xft = Xft.reshape([-1, 3])
            Xft = Xft[np.linalg.norm(Xft, axis=1) <= sz]
            sigma = np.array([sigmaZ, sigmaXY, sigmaXY])[np.newaxis]
            Xft_ = Xft / sigma
            norm_G = np.exp(-np.sum(Xft_ * Xft_, axis=-1) / 2.)
            norm_G = (norm_G - np.mean(norm_G)) / np.std(norm_G)
            im_centers__ = im_centers[3:].T.copy()
            im_centers__ = (im_centers__ - np.mean(im_centers__, axis=1)[:, np.newaxis]) / np.std(im_centers__, axis=1)[:, np.newaxis]
            hn = np.mean(im_centers__ * norm_G, axis=1)

        zc = np.sum(im_centers[0] * im_centers[3], axis=1) / np.sum(im_centers[3], axis=1)
        xc = np.sum(im_centers[1] * im_centers[3], axis=1) / np.sum(im_centers[3], axis=1)
        yc = np.sum(im_centers[2] * im_centers[3], axis=1) / np.sum(im_centers[3], axis=1)

        Xh = np.column_stack((zc, xc, yc, bk, a, habs, hn, h))

    if return_centers:
        return Xh, im_centers.T

    return Xh
  return Xh, np.array(im_centers)
return Xh
x]       im_centers_[3] -= bk
        a = np.sum
