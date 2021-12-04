"""
Provides the following bilateral methods:

* :func:`carli`
* :func:`jevons`
* :func:`dutot`
* :func:`laspeyres`
* :func:`paasche`
* :func:`geom_laspeyres`
* :func:`geom_paasche`
* :func:`drobish`
* :func:`marshall_edgeworth`
* :func:`palgrave`
* :func:`fisher`
* :func:`tornqvist`
* :func:`walsh`
* :func:`sato_vartia`
* :func:`geary_khamis_b`
* :func:`tpd`
* :func:`rothwell`
"""
import numpy as np

__author__ = ['Dr. Usman Kayani']

def carli(p0, p1):
    """Carli bilateral index, using price information.."""
    return np.mean(p1 / p0)

def jevons(p0, p1):
    """Jevons bilateral index, using price information.."""
    return np.prod((p1 / p0) ** (1 / len(p0)))

def dutot(p0, p1):
    """Dutot bilateral index, using price information.."""
    return np.sum(p1)/np.sum(p0)

def laspeyres(p0, p1, q0):
    """Laspeyres bilateral index, using price and base quantity information."""
    return lowe(p0, p1, q0)

def paasche(p0, p1, q1):
    """Paasche bilateral index, using price and current quantity information."""
    return lowe(p0, p1, q1)

def geom_laspeyres(p0, p1, q0):
    """Geometric Laspeyres bilateral index, using price and base quantity information."""
    s0 = (p0 * q0) / np.sum(p0 * q0)
    return np.prod((p1 / p0) ** s0)

def geom_paasche(p0, p1, q1):
    """Geometric Laspeyres bilateral index, using price and base quantity information."""
    s1 = (p1 * q1) / np.sum(p1 * q1)
    return np.prod((p1 / p0) ** s1)

def lowe(p0, p1, q):
    """Lowe bilateral index, using price and arbitrary quantity information."""
    return np.sum(p1 * q) / np.sum(p0 * q)

def young(p0, p1, p, q):
    """Young bilateral index, using price and arbitrary price and quantity information."""
    s = (p * q) / np.sum(p * q)
    return np.sum(s * (p1 / p0))

def drobish(p0, p1, q0, q1):
    """Drobish bilateral index, using price and quantity information."""
    return (laspeyres(p0, p1, q0) + paasche(p0, p1, q1)) / 2

def marshall_edgeworth(p0, p1, q0, q1):
    """Marshall-Edgeworth bilateral index, using price and quantity information."""
    return np.sum(p1 * (q0 + q1)) / np.sum(p0 * (q0 + q1))

def palgrave(p0, p1, q1):
    """Palgrave bilateral index, using price and arbitrary price and quantity information."""
    s1 = (p1 * q1) / np.sum(p1 * q1)
    return np.sum(s1 * (p1 / p0))

def fisher(p0, p1, q0, q1):
    """Fisher bilateral index, using price and quantity information."""
    return np.sqrt(laspeyres(p0, p1, q0) * paasche(p0, p1, q1))

def tornqvist(p0, p1, q0, q1):
    """Torqvist bilateral index, using price and quantity information."""
    s0 = (p0 * q0) / np.sum(p0 * q0)
    s1 = (p1 * q1) / np.sum(p1 * q1)
    return np.prod((p1 / p0) ** (0.5 * (s0 + s1)))

def walsh(p0, p1, q0, q1):
    """Walsh bilateral index, using price and quantity information."""
    return np.sum(np.sqrt(q0 * q1).dot(p1)) / np.sum(np.sqrt(q0 * q1).dot(p0))

def sato_vartia(p0, p1, q0, q1):
    """Sato-Vartia bilateral index, using price and quantity information."""
    s0 = (p0 * q0) / np.sum(p0 * q0)
    s1 = (p1 * q1) / np.sum(p1 * q1)
    w = (s1 - s0) / (np.log(s1) - np.log(s0))
    w = w / np.sum(w)
    return np.prod((p1 / p0) ** w)

def geary_khamis_b(p0, p1, q0, q1):
    """Geary-Khamis bilateral index, using price and quantity information."""
    hmean = (2 * q0 * q1) / (q0 + q1)
    return np.sum(hmean * p1) / np.sum(hmean * p0)

def rothwell(p0, p1, q0):
    """Rothwell bilateral index, using price and base quantity information."""
    pu = np.sum(p0 * q0) / np.sum(q0)
    return np.sum(p1 * q0) / np.sum(pu * q0)
