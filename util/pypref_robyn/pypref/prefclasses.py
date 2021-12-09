# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 11:10:47 2016

@author: Patrick Roocks
"""

import numpy as np

# for data sets
import pandas as pd

# Direkt import of c-compiled bnl
import bnl

from . import btg

# General preference classes 
# ==========================
 
# Exception class for all errors in this file
class PreferenceException(Exception):
  def __init__(self, value):
    self.value = value
  def __str__(self):
    return repr(self.value)

class pref:
  """  
  General main class for preferences.
  
  Consider `pypref.low` for constructing preferences and
  `pypref.pref.psel` for evaluating preferences (obtaining the optima). 
  """
  
  def get_str(self):
    return "(abstract preference)"
  def __str__(self):
    return "[Preference] " + self.get_str()
    
  # Operators for complex preferences, [pref] {*, &, |, +} [pref]
  def __mul__(self, other): return self.__get_cpx_pref(other, _pareto)
  def __and__(self, other): return self.__get_cpx_pref(other, _prior)
  def __or__ (self, other): return self.__get_cpx_pref(other, _intersect)
  def __add__(self, other): return self.__get_cpx_pref(other, _union)
    
  # Generate complex preference from operator
  def __get_cpx_pref(self, other, constructor):
   if (isinstance(self, empty)):
     return other
   elif (isinstance(other, empty)):
     return self
   else:
     return constructor(self, other)
     
  # Operator for reverse preference
  def __neg__(self): 
    return reverse(self)    
    
  # Preference evaluation
  # =====================
    

  def psel_idx(self, df, **top_args):
    """
    Performs a preference selection and returns the indices
    
    See `psel` for details.         
    
    """
  
    # Empty preference => Return entire data set
    if isinstance(self, empty): return(range(0, len(df)))
      
    # Empty data set => Return empty indices
    if len(df) == 0: return(np.array([], dtype = int))
    
    # Get second argument of score_vals (first argument is next_id)
    score_array = np.array(self.get_score_vals(df, 0)[1])
    
    # Serialize pref
    serial_pref = self.serialize()
    
    # pick top-k argument
    def get_arg(name, default = -1):
      if name in top_args.keys():
        return top_args[name]
      else:
        return default
    
    # get param and check for int parameter
    def get_int(name, default = -1):
      val = get_arg(name, default)
      if ((not isinstance(val, int)) or val < -1):
        raise PreferenceException('Parameter "' + name + '" must be a positive integer value')
      return val
        
    # get param and check for bool parameter
    def get_bool(name, default = False):
      val = get_arg(name, default)
      if (not isinstance(val, bool)):
        raise PreferenceException('Parameter "' + name + '" must be Boolean')
      return val
    
    if (len(top_args) > 0): # top-k mode
      topk          = get_int('top')
      at_least      = get_int('at_least')
      top_level     = get_int('top_level')
      and_connected = get_bool('and_connected', True)
      show_level    = get_bool('show_level', False)
  
      if topk == -1 and at_least == -1 and and_connected == -1:
        raise PreferenceException("Expected at least one topk parameter!")
        
      # Call Cython BNL
      res = bnl.runbnl_top(score_array, serial_pref, topk, at_least, top_level, and_connected)
      
      if show_level:
        # return tuple indices and level numbers
        return pd.DataFrame(np.transpose(np.array(res)), columns = ['_indices', '_level'])
      else:
        # just return the tuple indices
        return res[0]
      
    else: # usual mode (no top-k selection)
    
      # Call Cython BNL
      return bnl.runbnl(score_array, serial_pref)
  
  
  def psel(self, df, **top_args):  
    """
    Performs a preference selection and returns the tuples
    
    Parameters
    ----------
    
    df : data frame
      A data frame where the preference is applied to.
    
    topargs : One or more of the following arguments (optional):
      * A `top` value of k means that the k-best tuples of the data set are returned. 
        This may be non-deterministic, see below for details.
      * An `at_least` value of k returns the top-k tuples and additionally all tuples which are 
        not dominated by the worst tuple (i.e. the minima) of the Top-k set. 
        The number of tuples returned is greater or equal than
        `at_least`. In contrast to top-k, this is deterministic.
      * An `top_level` value of k returns all tuples from the k-best levels. 
        See below for the definition of a level.
      * The logical value `and_connected` is only relevant if more than one of the
        above `top`, `at_least`, `top_level` values are given.
        Then `True` returns the intersection of the different top-selections 
        and `False` returns the union.
        
    Returns
    -------
    
    A subset of `df` which is optimal for the given preference.
    If topargs are given, then a additional column `_level` is appended.
    
    Top-k Preference Selection
    ---------------------------
    
    For a given `top` value of k the k best elements and their level values are returned. 
    The level values are determined as follows:
    
    * All the maxima of a data set w.r.t. a preference have level 1.
    * The maxima of the remainder, i.e., the data set without the level 1 maxima, have level 2.
    * The n-th iteration of "Take the maxima from the remainder" leads to tuples of level n.
    
    """  
  
    if (len(top_args) > 0): # top-k selection
      
      if('show_level' in top_args.keys() and not top_args['show_level']): 
        # show_level was explicitly set to false ==> do not show level
        return df.iloc[self.psel_idx(df, **top_args)]
      else:
        # show level (true by default)
        top_args['show_level'] = True
        res = self.psel_idx(df, **top_args)
        # get tuples
        res_df = df.iloc[res['_indices']]
        # insert levels column
        res_df.insert(len(res_df.columns), '_level', np.array(res['_level']))
        return(res_df)
      
    else: # usual selection (no top-k)
      return df.iloc[self.psel_idx(df)]
    
    
  # Hasse diagramm / precedecessors / successors
  # ============================================    
  
  # Get the Hasse diagram as (n,2) int matrix containing all edges
  def btg(self, df):
    """
    Returns the Better-Than-Graph of the preference w.r.t. a given data set `df`, i.e.,
    an object of the class btg associated with the preference and the data set.
    
    The Better-Than-Graph contains information about predecessors and successors
    of tuples w.r.t. the preference. Additionally it can be visualized as a diagram
    using the GraphViz DOT interpreter.
    """
    return btg.btg(df, self)
    
    

# Special empty preference (neutral element for all complex preferences)
# cannot be evaluated
class empty(pref):
  def get_str(self):
    return "(empty)"


# Base preferences
# ================

class _basepref(pref):
  """
  Base preferences are used to describe the different goals 
  of a preference query. 

  
  Parameters
  ----------
  
  epxr : string or function
    Specifies either an expression over the data set where the preference should
    be applied to, or, a function operating on the data set.
    
  Returns
  -------
  
  A preference object. This can be used to retrieve the optimal elements w.r.t.
  the induced order of preference (see examples), or, to build a complex preference
  from it (see complex preferences, below examples).
    
  Details
  -------
  
  Mathematically, all base preferences are strict weak orders 
  (irreflexive, transitive and negative transitive).
  
  The three fundamental base preferences are:
  
  low("a"), high("a") : 
    Search for minimal/maximal values of a, i.e., 
    the induced order is the "smaller than" or "greater than" order 
    on the values of a. The values of a must be numeric values.
  true("a") : 
    Searches for true values in logical expressions, i.e., 
    TRUE is considered to be better than FALSE. The values of a must be
    logical values. For a tuplewise evaluation of a complex logical expression 
    one has to use the & and | operators for logical AND/OR 
    (and not the "or" and "and" operators).
    

  Examples
  --------
  
  The following two examples show two different, but semantically equivalent,
  ways to define a preference maximizing "4 * mpg + hp" for the mtcars data set.
  
  >>> pref = p.high("4 * mpg + hp")
  >>> pref.psel(p.get_mtcars())
  
  >>> pref = p.high(lambda x : 4 * x['mpg'] + x['hp'])
  >>> pref.psel(p.get_mtcars())
  
  The following example picks those cars having 4 cylinders and a miles-per-gallone
  value less then 23. If there would be no such cars, all cars would be returned.
  
  >>> p.true("(cyl == 4) & (mpg < 23.0)")
  >>> pref.psel(p.get_mtcars())
  
  
  Complex Preferences
  -------------------
  
  Base preferences and complex preferences can be combined with the follwing operators:
  
  * `p1 & p2` (Prioritization): Constructs the lexicgraphical order of `p1` and `p2`.
  * `p1 * p2` (Pareto): Constructs the Pareto preference (for Skyline queries) 
    involving the the preferences `p1` and `p2`.
  * `p1 | p2` (Intersection).
  * `p1 + p2` (Union).
  
  """

  eval_fun = None
  eval_str = None
  score_id = None  
  
  def __init__(self, expr):    
    if isinstance(expr, str):
      self.eval_str = expr
      self.eval_fun = None
    elif callable(expr):
      self.eval_fun = expr
      self.eval_str = None
    else:
      raise PreferenceException("Expected string or callable in base preference argument!")
      
  # Gat values from eval_str or eval_fun (not the actual score values, see low/high/true)
  def get_val(self, df):
    
    if (not isinstance(df, pd.core.frame.DataFrame)): 
      raise PreferenceException("Preference evaluation expects a DataFrame")

    if (self.eval_str != None):
      # Evaluate string and return array
      res = np.array(eval(self.eval_str, globals(), df), dtype = float)
      if np.isscalar(res): return np.array([res] * len(res), dtype = float)
      else:                return res
    elif (self.eval_fun != None):
      # Evaluate function over the entire data set
      return self.eval_fun(df)
      
  # String representation (without "[preference]")
  def inner_str(self):
    if (self.eval_str != None):
      return '"' + self.eval_str + '"'
    elif (self.eval_fun != None):
      return '[function]'
    else:
      return "(null)"
      
  # Return score values and save the id in the score data set
  def get_score_vals(self, df, next_id):
    self.score_id = next_id
    return (next_id + 1, self.calc_score(df))
      
  # Compare function NOT needed for BNL (implemented separately in C++) but for pred_succ functions      
      
  # Compare and Equality for all base prefs: if true, then t1 is better than t2
  def cmp(self, params):
    (t1, t2, score_lst) = params
    return score_lst[self.score_id][t1] < score_lst[self.score_id][t2]

  def eq(self, params):
    (t1, t2, score_lst) = params
    return score_lst[self.score_id][t1] == score_lst[self.score_id][t2]  
    
  def serialize(self):
    return [ord("s")]


class low(_basepref):  
  
  def get_str(self):
    return "low(" + self.inner_str() + ")"
  
  # Score for low is the identity
  def calc_score(self, df):
    return([self.get_val(df)])
    

class high(_basepref):
  
  def get_str(self):
    return "high(" + self.inner_str() + ")"
    
  # Score for high is the negated identity
  def calc_score(self, df):
    return([-self.get_val(df)])
  
  
class true(_basepref):
  
  def get_str(self):
    return "true(" + self.inner_str() + ")"
  
  # Score for true: 0 for True and 1 for False
  def calc_score(self, df):
    # Use 1.*(.) for conversion from bool to float
    return([1. - 1. * self.get_val(df)])
    
    
    
    
# Complex preferences
# ===================


class reverse(pref):
  """
  `reverse(p)` returns the converse of the preference `p`. 
  
  `-p` is a short-cut for `reverse(p)`.
  """

  p = None

  def __init__(self, _p):
    self.p = _p
    
  def operator(self):
    return "-"
    
  def get_score_vals(self, df, next_id):
    return self.p.get_score_vals(df, next_id)

  def get_str(self):
    return self.operator() + self.p.get_str()

  # Inherited equality
  def eq(self, params):
    return self.p.eq(params)
    
  # Compare function swaps the arguments
  def cmp(self, params):
    (t1, t2, score_lst) = params
    return self.p.cmp((t2, t1, score_lst))

  def serialize(self):
    return [ord(self.operator()), self.p.serialize()]


class _complexpref(pref):
  
  p1 = None
  p2 = None
  
  def __init__(self, _p1, _p2):
    self.p1 = _p1
    self.p2 = _p2
  
  def get_score(self, df):
    return self.p1.get_score(df) + self.p2.get_score(df)

  def get_str(self):
    return self.p1.get_str() + " " + self.operator() + " " + self.p2.get_str()
    
  def get_score_vals(self, df, next_id):
    (new_id1, score_list1) = self.p1.get_score_vals(df, next_id)
    (new_id2, score_list2) = self.p2.get_score_vals(df, new_id1)
    return (new_id2, score_list1 + score_list2)
  
  # Equality for all complex preferences (Compare is preference-specific)
  def eq(self, params):
    return self.p1.eq(params) & self.p2.eq(params)
    
  def serialize(self):
    return [ord(self.operator()), self.p1.serialize(), self.p2.serialize()]




class _pareto(_complexpref):
  
  def operator(self): return "*"
  
  # use | and & instead of and/or such that these functions operate pointwise for pred/succ functions
  def cmp(self, params):
    return ( ((self.p1.cmp(params) | self.p1.eq(params)) & self.p2.cmp(params)) | 
             ((self.p2.cmp(params) | self.p2.eq(params)) & self.p1.cmp(params))   )              
  

class _prior(_complexpref):
  
  def operator(self): return "&"
  
  def cmp(self, params):
    return self.p1.cmp(params) | (self.p1.eq(params) & self.p2.cmp(params))
  
  
class _intersect(_complexpref):
  
  def operator(self): return "|"
  
  def cmp(self, params):
    return (self.p1.cmp(params) & self.p2.cmp(params))
  
  
class _union(_complexpref):
  
  def operator(self): return "+"  

  def cmp(self, params):
    return (self.p1.cmp(params) | self.p2.cmp(params))
           