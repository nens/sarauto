### sarauto
 create vegetable maps from radar satellite data


## Installation

On Linux (Ubuntu 18)::
 
 $ sudo apt-get install libgdal-dev

Instal SNAP engine::

 $ sudo apt install default-jre 
 $ wget http://step.esa.int/downloads/7.0/installers/esa-snap_sentinel_unix_7_0.sh
 $ sh ./esa-snap_sentinel_unix_7_0.sh

Make the python virtualenv::

 $ virtualenv .venv --python=python2
 $ source .venv/bin/activate
 $ pip install -c constraints.txt numpy scipy spectral tifffile pygdal==2.2.3.* statsmodels tqdm numba joblib
 
Configure SNAP engine in python::

 $ <snap install dir>/bin/snappy-conf /home/<fill in sarauto repo dir>/.venv/bin/python
 $ ln -s ~/.snap/snap-python/snappy .venv/local/lib/python2.7/site-packages/snappy
