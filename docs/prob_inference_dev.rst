prob\_inference\_dev
=======================================

.. fuck this it doesn't work
.. .. automodule:: prob_inference_dev
..   :members:

.. automodule:: prob_inference_dev
    :noindex:
    
.. :synopsis: Cython module for inner loop methods for :mod:`prob_inference` in a :class:`formula_gen.PBModel`.

Typedefs::

   ctypedef np.int_t DINT_t
   ctypedef np.float_t DFLOAT_t
   
Variables:

N = number of observations on a plate (see :class:`formula_gen.PBPlate`).

M = number of propositional variables (see :class:`formula_gen.PBPlate`).

NNodes = number of nodes in a bdd (see :class:`knowledge_compilation.BDD`).
   
.. py:function:: seed_gsl_cy(unsigned int seed)

   Seed the GNU scientific library (gsl) random number generator (rng).
   
   :param seed: seed for rng
   :type seed: unsigned int
   
.. py:function:: backward_plate_cy(np.ndarray[DINT_t, ndim=2]      bdd,  np.ndarray[DINT_t, ndim=3]      plate, np.ndarray[DFLOAT_t, ndim=2]    prob)
    :module: prob_inference_dev
    
    Compute backward probability for a set of identical BDDs (with different parameters).
    
    :param bdd:
    :type bdd: see :class:`knowledge_compilation.BDD`
    :param plate:
    :type plate: see :class:`formula_gen.PBPlate`
    :rtype: ndarray of shape [N, NNodes, 2]
    
.. py:function:: sample_bdd_plate_cy(    np.ndarray[DINT_t, ndim=3]      plate,    np.ndarray[DINT_t, ndim=2]      bdd,    np.ndarray[DINT_t, ndim=1]      reps,    np.ndarray[DFLOAT_t, ndim=3]    beta,    np.ndarray[DFLOAT_t, ndim=2]    prob,                                    x # TODO optimize to array    )
    
    Sample a set of BDDs and update x. Sampling x means:
    
    1. :func:`backward_plate_cy`
    2. :func:`sample_bdd_plate_cy`
    
    :param plate:
    :type plate: see :class:`formula_gen.PBPlate`
    :param bdd:
    :type bdd: see :class:`knowledge_compilation.BDD`
    :param reps:
    :type reps: see :class:`formula_gen.PBPlate`
    :param beta:
    :type beta: returned by :func:`backward_plate_cy`
    :param prob: probabilities of the propositional variables in the BDDs
    :type prob: returned by :func:`reparam_cy` (spoiler NXM ndarray)
    :param x: datastructure for x
    :type x: list of NDist ndarrays, each of shape n_vars X n_cat (see :class:`formula_gen.PBModel`)
    :rtype: None
    
.. py:function:: reparam_cy(    cat_list,    theta,    np.ndarray[DINT_t, ndim=3] plate,    np.ndarray[DINT_t, ndim=2] kid)

    Compute the parameters of the propositional variables in the BDDs based on theta.
    
    :param cat_list: 
    :type cat_list: see :class:`formula_gen.PBPlate`
    :param theta:
    :type theta: list of NDist ndarrays, each of shape n_vars X n_cat (see :class:`formula_gen.PBModel`)
    :param plate:
    :type plate:  see :class:`formula_gen.PBPlate`
    :param kid:
    :type kid: see :class:`formula_gen.PBPlate`
    :rtype: ndarray of shape N X M    
    
