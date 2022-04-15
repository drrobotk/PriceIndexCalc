"""
Provides the following bilateral methods:

* :func:`carli`
* :func:`jevons`
* :func:`dutot`
* :func:`lowe`
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
* :func:`young`
"""
import numpy as np

__author__ = ['Dr. Usman Kayani']

def carli(
    p0: np.array, 
    p1: np.array,
) -> float:
    """
    Carli bilateral index, using price information.
    
    .. math::
        \\text{Carli} = \\frac{\\sum_{i=1}^{n} p_i}{\\sum_{i=1}^{n} p_0}

    :param p0: Base price vector.
    :param p1: Current price vector.
    """
    return np.mean(p1 / p0)

def jevons(
    p0: np.array, 
    p1: np.array
) -> float:
    """
    Jevons bilateral index, using price information.
    
    .. math::
        \\text{Jevons} = \\frac{\\sum_{i=1}^{n} p_i}{\\sum_{i=1}^{n} p_0}

    :param p0: Base price vector.
    :param p1: Current price vector.
    """
    return np.prod((p1 / p0) ** (1 / len(p0)))

def dutot(
    p0: np.array, 
    p1: np.array
) -> float:
    """
    Dutot bilateral index, using price information.
    
    .. math::
        \\text{Dutot} = \\frac{\\sum_{i=1}^{n} p_i}{\\sum_{i=1}^{n} p_0}

    :param p0: Base price vector.
    :param p1: Current price vector.
    """
    return np.sum(p1)/np.sum(p0)

def laspeyres(
    p0: np.array, 
    p1: np.array,
    q0: np.array,
) -> float:
    """
    Laspeyres bilateral index, using price and base quantity information.
    
    .. math::
        \\text{Laspeyres} = \\frac{\\sum_{i=1}^{n} p_i}{\\sum_{i=1}^{n} p_0}

    :param p0: Base price vector.
    :param p1: Current price vector.
    :param q0: Base quantity vector.
    """
    return lowe(p0, p1, q0)

def paasche(
    p0: np.array, 
    p1: np.array,
    q1: np.array,
) -> float:
    """Paasche bilateral index, using price and current quantity information."""
    return lowe(p0, p1, q1)

def geom_laspeyres(
    p0: np.array,
    p1: np.array,
    q0: np.array,
) -> float:
    """
    Geometric Laspeyres bilateral index, using price and base quantity information.
    
    .. math::
        \\text{Geometric Laspeyres} = \\frac{\\sum_{i=1}^{n} p_i}{\\sum_{i=1}^{n} p_0}

    :param p0: Base price vector.
    :param p1: Current price vector.
    :param q0: Base quantity vector.
    """
    s0 = (p0 * q0) / np.sum(p0 * q0)
    return np.prod((p1 / p0) ** s0)

def geom_paasche(
    p0: np.array, 
    p1: np.array,
    q1: np.array,
) -> float:
    """
    Geometric Paasche bilateral index, using price and base quantity information.

    .. math::
        \\text{Geometric Paasche} = \\frac{\\sum_{i=1}^{n} p_i}{\\sum_{i=1}^{n} p_0}

    :param p0: Base price vector.
    :param p1: Current price vector.
    :param q1: Base quantity vector.
    """
    s1 = (p1 * q1) / np.sum(p1 * q1)
    return np.prod((p1 / p0) ** s1)

def lowe(
    p0: np.array, 
    p1: np.array,
    q: np.array,
) -> float:
    """
    Lowe bilateral index, using price and arbitrary quantity information.
    
    .. math::
        \\text{Lowe} = \\frac{\\sum_{i=1}^{n} p_i}{\\sum_{i=1}^{n} p_0}

    :param p0: Base price vector.
    :param p1: Current price vector.
    :param q: Arbitrary quantity vector.
    """
    return np.sum(p1 * q) / np.sum(p0 * q)

def young(
    p0: np.array,
    p1: np.array,
    p: np.array, 
    q: np.array,
) -> float:
    """
    Young bilateral index, using price and arbitrary price and quantity information.
    
    .. math::

        \\text{Young} = \\frac{\\sum_{i=1}^{n} p_i}{\\sum_{i=1}^{n} p_0}

    :param p0: Base price vector.
    :param p1: Current price vector.
    :param p: Arbitrary price vector.
    :param q: Arbitrary quantity vector.
    """
    s = (p * q) / np.sum(p * q)
    return np.sum(s * (p1 / p0))

def drobish(
    p0: np.array, 
    p1: np.array,
    q0: np.array,
    q1: np.array,
) -> float:
    """
    Drobish bilateral index, using price and quantity information.
    
    .. math::
        \\text{Drobish} = \\frac{\\sum_{i=1}^{n} p_i}{\\sum_{i=1}^{n} p_0}

    :param p0: Base price vector.
    :param p1: Current price vector.
    :param q0: Base quantity vector.
    :param q1: Current quantity vector.
    """
    return (laspeyres(p0, p1, q0) + paasche(p0, p1, q1)) / 2

def marshall_edgeworth(
    p0: np.array,
    p1: np.array,
    q0: np.array,
    q1: np.array,
) -> float:
    """
    Marshall-Edgeworth bilateral index, using price and quantity information.
    
    .. math::
        \\text{Marshall-Edgeworth} = \\frac{\\sum_{i=1}^{n} p_i}{\\sum_{i=1}^{n} p_0}

    :param p0: Base price vector.
    :param p1: Current price vector.
    :param q0: Base quantity vector.
    :param q1: Current quantity vector.
    """
    return np.sum(p1 * (q0 + q1)) / np.sum(p0 * (q0 + q1))

def palgrave(
    p0: np.array,
    p1: np.array,
    q1: np.array,
) -> float:
    """
    Palgrave bilateral index, using price and arbitrary price and quantity information.
    
    .. math::
        \\text{Palgrave} = \\frac{\\sum_{i=1}^{n} p_i}{\\sum_{i=1}^{n} p_0}

    :param p0: Base price vector.
    :param p1: Current price vector.
    :param q1: Arbitrary quantity vector.
    """
    s1 = (p1 * q1) / np.sum(p1 * q1)
    return np.sum(s1 * (p1 / p0))

def fisher(
    p0: np.array, 
    p1: np.array,
    q0: np.array,
    q1: np.array,
) -> float:
    """
    Fisher bilateral index, using price and quantity information.
    
    .. math::
        \\text{Fisher} = \\frac{\\sum_{i=1}^{n} p_i}{\\sum_{i=1}^{n} p_0}

    :param p0: Base price vector.
    :param p1: Current price vector.
    :param q0: Base quantity vector.
    """
    return np.sqrt(laspeyres(p0, p1, q0) * paasche(p0, p1, q1))

def tornqvist(
    p0: np.array,
    p1: np.array,
    q0: np.array,
    q1: np.array,
) -> float:
    """
    Torqvist bilateral index, using price and quantity information.
    
    .. math::
        \\text{Torqvist} = \\frac{\\sum_{i=1}^{n} p_i}{\\sum_{i=1}^{n} p_0}

    :param p0: Base price vector.
    :param p1: Current price vector.
    :param q0: Base quantity vector.
    :param q1: Current quantity vector.
    """
    s0 = (p0 * q0) / np.sum(p0 * q0)
    s1 = (p1 * q1) / np.sum(p1 * q1)
    return np.prod((p1 / p0) ** (0.5 * (s0 + s1)))

def walsh(
    p0: np.array, 
    p1: np.array,
    q0: np.array,
    q1: np.array,
) -> float:
    """
    Walsh bilateral index, using price and quantity information.
    
    .. math::
        \\text{Walsh} = \\frac{\\sum_{i=1}^{n} p_i}{\\sum_{i=1}^{n} p_0}

    :param p0: Base price vector.
    :param p1: Current price vector.
    :param q0: Base quantity vector.
    :param q1: Current quantity vector.
    """
    return np.sum(np.sqrt(q0 * q1).dot(p1)) / np.sum(np.sqrt(q0 * q1).dot(p0))

def sato_vartia(
    p0: np.array,
    p1: np.array,
    q0: np.array,
    q1: np.array,
) -> float:
    """
    Sato-Vartia bilateral index, using price and quantity information.
    
    .. math::
        \\text{Sato-Vartia} = \\frac{\\sum_{i=1}^{n} p_i}{\\sum_{i=1}^{n} p_0}

    :param p0: Base price vector.
    :param p1: Current price vector.
    :param q0: Base quantity vector.
    :param q1: Current quantity vector.
    """
    s0 = (p0 * q0) / np.sum(p0 * q0)
    s1 = (p1 * q1) / np.sum(p1 * q1)
    w = (s1 - s0) / (np.log(s1) - np.log(s0))
    w = w / np.sum(w)
    return np.prod((p1 / p0) ** w)

def geary_khamis_b(
    p0: np.array,
    p1: np.array,
    q0: np.array,
    q1: np.array,
) -> float:
    """
    Geary-Khamis bilateral index, using price and quantity information.
    
    .. math::
        \\text{Geary-Khamis} = \\frac{\\sum_{i=1}^{n} p_i}{\\sum_{i=1}^{n} p_0}

    :param p0: Base price vector.
    :param p1: Current price vector.
    :param q0: Base quantity vector.
    :param q1: Current quantity vector.
    """
    hmean = (2 * q0 * q1) / (q0 + q1)
    return np.sum(hmean * p1) / np.sum(hmean * p0)

def rothwell(
    p0: np.array,
    p1: np.array,
    q0: np.array,
) -> float:
    """
    Rothwell bilateral index, using price and base quantity information.
    
    .. math::
        \\text{Rothwell} = \\frac{\\sum_{i=1}^{n} p_i}{\\sum_{i=1}^{n} p_0}

    :param p0: Base price vector.
    :param p1: Current price vector.
    :param q0: Base quantity vector.
    """
    pu = np.sum(p0 * q0) / np.sum(q0)
    return np.sum(p1 * q0) / np.sum(pu * q0)
