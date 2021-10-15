# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 11:14:18 2016

@author: Patrick Roocks
"""

import os
import numpy as np
import subprocess

# for reduce
import functools as ft

# Direkt import of c-compiled bnl
import bnl


# Exception class for all errors in this file
class BTGException(Exception):
  def __init__(self, value):
    self.value = value
  def __str__(self):
    return repr(self.value)

# Config
# ======

dot_path = "C:/Program Files (x86)/Graphviz2.38/bin/dot.exe"
viewer_path = "C:/Program Files (x86)/IrfanView/i_view32.exe"

# Set paths
def set_paths(dot, viewer):
  """ 
  Set paths for GraphViz DOT interpreter and PNG viewer, used for the 
  show_graph method of the btg class.
  
  
  Parameters
  ----------

  dot : string
    Absolute path to DOT interpreter  
    
  viewer : string
    Absolute path to PNG Viewer
  """
  global dot_path, viewer_path
  dot_path = dot
  viewer_path = viewer


class btg:
  """
  Better-Than-Graph for preferences, containing the Sucessor/Predecessor functions
  and visualiations of Better-Than-Graphs
  
  Use pref.btg(df) to create a Better-Than-Graph.
  
  """

  
  p = None # preference  
  df = None # DataFrame
  scores = None # score array for cmp/eq functions
  
  # Matrix containing the Hasse diagram
  hasse_diag = np.array([[]]) # empty matrix
  
  # Indices of maxima
  max_indices = np.array([])
  
  # Options for plotting
  fillcolors = None  
  fontcolors = None
  labels = None
  flip_edges = False
  
  # Legend as DOT-str (subgraph!)
  legend = ""
  
  def __init__ (self, df, pref):
    # Store pref/df
    self.p = pref
    self.df = df    
    
    # Calculate the Hasse diagram
    self.scores = np.array(pref.get_score_vals(df, 0)[1])
    self.hasse_diag = np.transpose(bnl.get_transitive_reduction(self.scores, pref.serialize()))
    
    # Calculate maxima
    self.max_indices = pref.psel_idx(df)
    
    # Set default options
    self.labels = [str(x) for x in self.df.index]
    
    
  # Generate DOT code
  # -----------------
  
  # Colorize predecessor/successor nodes
  def add_color_gradient(self, center, up_limit = -1, down_limit = -1):
    """
    Adds colors to predecessors/successors for a given subset. 
    The colors of the other nodes are fading to white with growing distance from the center.
    
    Parameters
    ----------
    
    center : List of indices
      Specifies the set of nodes for which predecessors and successors are shown.
      This set must by an antichain set for the preference, i.e., an invariant w.r.t.
      the preference selection.
    
    up_limit : Integer
      Maximal number of colored predecessors of `center`.
      If this value equals `-1`,
      all nodes from `center` to the top of the graph are colored
      
    down_limit : Integer
      Maximal number of colored successors of `center`.
      If this value equals `-1`,
      all nodes from `center` to the top of the graph are colored
    
    Returns
    -------
    
    Nothing. All modifications are written to the btg object. 
    Consider the functions `dot_str`, `save_dot` and `show_graph`
    to get the graph output.
    
    """
    
    # Check if `center` is antichain
    if len(self.p.psel_idx(self.df.iloc[center])) != len(center):
       raise BTGException("The set `center` must be antichain. " +
         "Ensure that `btg.p.psel_idx(btg.df.iloc[center])` is equal to `center`.")
    
    
    up_color   = np.array([0,0.5,1])  # blue
    down_color = np.array([0,0.75,0]) # green
     
    levels_up_max = 0
    levels_down_max = 0
    up_val = center
    down_val = center
    finished_down = False
    finished_up = False
    
    # Functions to retrieve local minima/maxima for every successor/predecessor step
    # Note that all steps must correspond to antichain sets    

    def get_local_maxima(indices):
      loc_ind = self.p.psel_idx(self.df.iloc[indices])
      return(indices[loc_ind])
      
    def get_local_minima(indices):
      loc_ind = (-self.p).psel_idx(self.df.iloc[indices])
      return(indices[loc_ind])
  
    # calculate levels_up_max / levels_down_max
    while(not (finished_up and finished_down)):
      # Take minimal/maximal nodes in predecessor/successor sets
      up_val   = get_local_minima(self.hasse_pred(up_val))
      down_val = get_local_maxima(self.hasse_succ(down_val))
      # Check if downward/upward search terminates
      if (len(up_val)   == 0 or up_limit   == levels_up_max):   finished_up = True
      if (len(down_val) == 0 or down_limit == levels_down_max): finished_down = True
      if not finished_up:   levels_up_max += 1
      if not finished_down: levels_down_max += 1
    
    # colors are string with maximally 20 chars
    fillcolors = np.array(['white'] * len(self.df), dtype="<U20")
    fillcolors[center] = 'orange'
    
    # convex combination of colors and hex output
    def mix_color(coef, col):
      base_color = np.array([1,1,1])
      col = col * (1 - coef) + base_color * coef
      return '"#' + ''.join(['%02X' % int(c*255) for c in col]) + '"'

    # Init legend colors/labels
    num_leg = 1 + levels_up_max + levels_down_max
    leg_labels = np.array([''] * num_leg, dtype="<U30")
    leg_colors = np.array([''] * num_leg, dtype="<U20")
    leg_labels[levels_up_max] = 'Selected nodes'
    leg_colors[levels_up_max] = 'orange'
    

    # Assign colors
    levels_up = 0   if levels_up_max > 0   else 1 # Avoid up-branch if no up-levels
    levels_down = 0 if levels_down_max > 0 else 1 # Avoid down-branch if no up-levels
    up_val = center
    down_val = center
    while(levels_up <= levels_up_max or levels_down <= levels_down_max):
      up_val   = get_local_minima(self.hasse_pred(up_val))
      down_val = get_local_maxima(self.hasse_succ(down_val))
      levels_up += 1
      levels_down += 1
      if (levels_up <= levels_up_max): 
        col = mix_color((levels_up - 1) / levels_up_max, up_color)
        # Colors for nodes
        fillcolors[up_val] = col
        # Colors for legend - levels_up below center
        leg_colors[levels_up_max - levels_up] = col
        # Label for legend
        leg_labels[levels_up_max - levels_up] = "level -" + str(levels_up)
      if (levels_down <= levels_down_max): 
        col = mix_color((levels_down - 1) / levels_down_max, down_color)
        # Colors for nodes
        fillcolors[down_val] = col
        # Colors for legend
        leg_colors[levels_up_max + levels_down] = col
        # Label for legend
        leg_labels[levels_up_max + levels_down] = "level +" + str(levels_down)
    
    self.fillcolors = fillcolors
    
    # Init legend
    legend = ('subgraph cluster_01 {\n' + 
              'mindist=0;\n' +
              'ranksep=0;\n' + 
              'nodesep=0;\n' + 
              'node[shape=box,margin="0,0",width=1.5, height=0.5];\n' +
              'edge[style=invis];\n' + 
              '"leghead" [label="Legend",color=white, fontsize=20];\n')
              
    legend += ''.join(['"leg%i" [label="%s",style=filled,fillcolor=%s];\n' 
                       % (i, leg_labels[i], col) for i, col in enumerate(leg_colors)])
              
    legend += 'leghead -> leg0;\n'
    legend += ''.join(['leg%i -> leg%i;' % (i, i+1) for i in range(num_leg-1)])              
              
    self.legend = legend + '}\n'


  def dot_str(self):    
    """
    Returns the graph as a string in the GraphViz DOT language
    """
    
    # Init Graph
    res_str = "digraph G {\n"
    
    if self.flip_edges:
      res_str += "rankdir = BT\n"
    
    # Maxima as first layer
    res_str += ("{\nrank=same;\n" +
                ''.join(["%i;\n" % t for t in self.max_indices]) +
                "}\n")

    # Labels
    if self.fillcolors is None and self.fontcolors is None:
      label_str = ''.join(['"%i" [label="%s"]\n' % (i, l) for i, l in enumerate(self.labels)])
    else:
      if self.fillcolors is None: self.fillcolors = ['white'] * len(self.df)
      if self.fontcolors is None: self.fontcolors = ['black'] * len(self.df)
      label_str = ''.join(['"%i" [label="%s", style=filled, fillcolor=%s, fontcolor=%s]\n' 
                           % (i, l, self.fillcolors[i], self.fontcolors[i]) for i, l in enumerate(self.labels)])

    res_str += label_str

    # Indices for FROM and TO edge parameters
    le_edge = 1 if self.flip_edges else 0
    ri_edge = 0 if self.flip_edges else 1

    # Edges
    res_str += ''.join(["%i -> %i;\n" % (x[le_edge], x[ri_edge]) for x in self.hasse_diag])
      
    # Add legend (if existing, may be an empty string)
    res_str += self.legend + "\n"
      
    # Finalize Graph and return
    return res_str + "}"


  # Save/launch
  # -----------

  def save_dot(self, file):
    """
    Saves the graph as file in the GraphViz DOT format
    
    Parameters
    ----------
    
    file : String
      Output path for the DOT file
    """
    
    f = open(file, 'w')
    f.write(self.dot_str())
    f.close()
    return 0


  def show_graph(self, **args):
    """
    Saves the graph as file in the GraphViz DOT format, 
    calls the DOT interpreter to produce a PNG image,
    and finally calls a image viewer to show the generated PNG image.
    
    Parameters (optional)
    ---------------------
    
    labels : list/array of strings
      Labels for the graph nodes

    fillcolors : list/array of strings with color values
      Background colors for the graph nodes
    
    fontcolors : list/array of strings with color values
      Font colors for the graph nodes    
    
    flip_edges : Boolean
      By default, the edges in the directed graph point from better to
      worse nodes. If `flip_edges` is set to True, then the edges point
      from worse to better nodes.
      
    Returns
    -------
    
    Nothing. Calls subprocesses for the DOT interpreter and Image Viewer
    
    See also
    --------
    
    Consider `set_paths` to specify the paths to DOT and the image viewer.
    
    
    """
    
    # Check paths
    err = ''
    if not os.path.exists(dot_path):
      err = 'Path to GraphViz DOT (' + dot_path + ') interpreter is not valid. '
    if not os.path.exists(viewer_path):
      err += 'Path to PNG Viewer (' + viewer_path + ') is not valid. '
    
    if err != '':
      err += 'Use `pypref.set_paths` to assign valid paths.'
      raise BTGException(err)
    
    # Take arguments
    if ('labels' in args): self.labels = args['labels']
    if ('fillcolors' in args): self.fillcolors = args['fillcolors']
    if ('fontcolors' in args): self.fontcolors = args['fontcolors']
    if ('flip_edges' in args): self.flip_edges = args['flip_edges']
        
    # Write file
    self.save_dot("tmpgraph.dot")
  
    # Create graph image and wait to complete  
    subprocess.call([dot_path, "-Tpng", "tmpgraph.dot", "-o", "tmpgraph.png"])  
    
    # Open viewer
    subprocess.Popen([viewer_path, "tmpgraph.png"])
    
    return 0
    
  # Predecessor/Sucessor functions
  # ------------------------------

  # Get predecessors/successors (if succ==True) of all vals. Return union of them if intersect==False
  def __hasse_predsucc(self, val, intersect, succ):
    
    # Intersection corresponds to logical and, union to logical or
    reduce_fun = (lambda x,y: x & y) if intersect else (lambda x,y: x | y)

    # Hasse diagram columns
    l_index = 0 if succ else 1
    r_index = 1 if succ else 0
    
    hd = self.hasse_diag
     
    if len(val) == 0:
      return np.array([])
    else:
      # Find all predecessors/successors in Hasse diagram using logical indexing
      hd_row_indices = ft.reduce(reduce_fun, [hd[:,l_index] == x for x in val])
      # As nodes may have multiple endpoints, we have to delete duplicates
      return np.unique(hd[hd_row_indices, r_index])
      
  def hasse_pred(self, val, intersect = False):
    """
    Returns the direct predecessors of val.
    
    Parameters
    ----------
    
    val : list/array of integers
      The indices for which the predecessors are requested.
    
    intersect : Boolean
      If this is False (by default) the union of the direct predecessors of `val` is returned.
      For `intersect = TRUE` the intersection of those values is returned.
      
    See also
    ---------      
    
    hasse_succ, all_pred, all_succ
      
    """
    
    return self.__hasse_predsucc(val, intersect, succ = False)
    
  def hasse_succ(self, val, intersect = False):
    """
    Returns the direct successors of val.
    
    Parameters
    ----------
    
    val : list/array of integers
      The indices for which the succesors are requested.
    
    intersect : Boolean
      If this is False (by default) the union of the direct successors of `val` is returned.
      For `intersect = TRUE` the intersection of those values is returned.
      
    See also
    ---------      
    
    hasse_pred, all_pred, all_succ
      
    """
    return self.__hasse_predsucc(val, intersect, succ = True)
      

  def __all_predsucc(self, val, intersect, succ):
    
    # Intersection corresponds to logical and, union to logical or
    reduce_fun = (lambda x,y: x & y) if intersect else (lambda x,y: x | y)
    
    # All indices of the data set
    all_inds = np.array(range(len(self.df)))

    # Compute list of predecessors / successors    
    if succ:
      lst = [self.p.cmp((x, all_inds, self.scores)) for x in val]    
    else:
      lst = [self.p.cmp((all_inds, x, self.scores)) for x in val]    
      
    # Reduce and return indices
    return all_inds[ft.reduce(reduce_fun, lst)]
  
  def all_pred(self, val, intersect = False):
    """
    Returns all predecessors of val, i.e., predecessors w.r.t. the transitive closure.
    
    Parameters
    ----------
    
    val : list/array of integers
      The indices for which the predecessors are requested.
    
    intersect : Boolean
      If this is False (by default) the union of all predecessors of `val` is returned.
      For `intersect = TRUE` the intersection of those values is returned.
      
    See also
    ---------      
    
    hasse_pred, hasse_succ, all_succ
      
    """
    return self.__all_predsucc(val, intersect, succ = False)
    
  def all_succ(self, val, intersect = False): 
    """
    Returns all successors of val, i.e., successors w.r.t. the transitive closure.
    
    Parameters
    ----------
    
    val : list/array of integers
      The indices for which the successors are requested.
    
    intersect : Boolean
      If this is False (by default) the union of all successors of `val` is returned.
      For `intersect = TRUE` the intersection of those values is returned.
      
    See also
    ---------      
    
    hasse_pred, hasse_succ, all_pred
      
    """
    return self.__all_predsucc(val, intersect, succ = True)
      
# ------------------------------------------------------------------------------------------------
  