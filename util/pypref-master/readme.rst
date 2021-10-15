Database Preferences and Skyline Computation in Python
======================================================

Routines to select and visualize the maxima for a given strict partial 
order. This especially includes the computation of the Pareto 
frontier, also known as (Top-k) Skyline operator, and some 
generalizations (database preferences).

This the Python port of the rPref package.

Package web site: http://p-roocks.de/rpref/index.php?section=pypref

Install and run
---------------

Copy the repository, e.g., in a sub folder of your home directory

::

  cd ~
  git clone https://github.com/patrickroocks/pypref pypref

  
Start Python, move to this directory (replace "/home/patrick" by your home directory) and import it:
  
::
  
  import os
  os.chdir("/home/patrick/pypref")
  import pypref as p
  
If everything went well, the following first tiny example should work:

::

  mtcars = p.get_mtcars()
  pref = p.high("mpg") * p.high("hp")
  sky = pref.psel(mtcars)
  btg = pref.btg(mtcars)

See pypref-examples.py for more examples.