peircebayes
=============

``peircebayes`` (referred to as PB in this documentation) is a probabilistic logic programming language designed for inference in probabilistic models with Dirichlet priors. See [1]_ for more details.

Installation
-------------

PB is implemented in Prolog and Python (2.7).

*Python core dependencies*

.. code-block:: rst

    numpy>=1.9.2
    scipy>=0.15.1
    matplotlib>=1.4.3
    cython>=0.22.1

*Other dependencies*

For installation, see the hyperlinks of the dependecies.

The Prolog implementation used is yap_. A standard YAP installation is fine.

The code for sampling uses the GNU scientific library ( gsl_ ).

pyCUDD_ is used for compiling BDDs in Python. If you're feeling lucky, try the steps below (otherwise see source files):

1. in the cudd folder (make sure the GCC flags are appropriately set in Makefile): 

.. code-block:: sh

    make clean
    make
    make libso

2. in the pycudd folder:

.. code-block:: sh

    make depend
    make


3. in your configuration file (e.g. ``~/.bashrc``) (replace the path to pycudd2.0.2 with your own):

.. code-block:: sh

    export PYTHONPATH=$PYTHONPATH:~/Programs/pycudd2.0.2/pycudd
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/Programs/pycudd2.0.2/cudd-2.4.2/lib

How to cite
-------------

For the moment, please cite PB using bibtex:

.. code-block:: tex

    @misc{Turliuc15,
        title = {peircebayes - Probabilistic Abductive Logic Programming using Dirichlet Priors},
        author = {Calin Rares Turliuc},
        url = {http://github.com/raresct/peircebayes}
    }

A paper describing PB has been accepted at the Second Workshop on Probabilistic Logic Programming. We will update the citation once we have the bibtex reference to the paper.

Contact
-------------

If you have any questions/problems/suggestions, don't hesitate to contact me at: ``raresturliuc [at] gmail [dot] com``. Note that my willingness to help with installation issues is significantly reduced on OSs different from Ubuntu 14.10.

References
-------------

.. [1] Turliuc, C.R., Dickens, L., Russo, A., Broda, K., *Probabilistic Abductive Logic Programming using Dirichlet Priors*, Second Workshop on Probabilistic Logic Programming, 2015, to appear.
.. _pyCUDD: http://bears.ece.ucsb.edu/pycudd.html
.. _yap: http://www.dcc.fc.up.pt/~vsc/Yap/index.html
.. _gsl: http://www.gnu.org/software/gsl/

