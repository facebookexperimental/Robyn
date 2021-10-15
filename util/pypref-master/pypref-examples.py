# -*- coding: utf-8 -*-
"""
Example file for the pypref package
"""


# Small pypref examples
# =====================

# include package
import pypref as p

# include matplotlib
import matplotlib.pyplot as plt


# Skyline plot
# ------------

# load mtcars data set given in pypref (motor trends data set from R)
mtcars = p.get_mtcars()

# preference for cars with minimal fuel consumption (high mpg value) and high power
pref = p.high("mpg") * p.high("hp")

def plot_skyline(dataset, pref):
  # plot all points
  plt.plot(dataset['mpg'], dataset['hp'], 'bo', fillstyle="none")
  
  # select optimal cars according to this preference (skyline)
  sky = pref.psel(dataset)

  # highlight skyline
  plt.plot(sky['mpg'], sky['hp'], 'bo')
  
  # show plot
  plt.show()
  
plot_skyline(mtcars, pref)



# Level value plot
# ----------------

# plot all levels and the Pareto frontier of each level
def plot_levels(dataset, pref):

  # get level values for all tuples from the data set
  res = pref.psel(dataset, top = len(dataset))
  
  # plot each level front line in a different color
  for level in range(1, res['_level'].max() + 1):
    pts = res.loc[res['_level'] == level].sort_values("mpg")
    plt.step(pts['mpg'], pts['hp'], 'o', label = "Level " + str(level))

  # show legend and plot
  plt.legend()
  plt.show()
  
# show level plot for data set and preference as given above
plot_levels(mtcars, pref)



# BTG examples
# ------------

# Create BTG (Better-Than-Graph) showing all the Better-Than-Relations
# for the preference and data set from above
btg = pref.btg(mtcars)

# create "cyl;mpg" labels
labels = [str(x['cyl']) + "; " + str(x['mpg']) for i, x in mtcars.iterrows()]

# Set paths to GraphViz and Viewer
# See http://www.graphviz.org/ to get GraphViz
p.set_paths(dot = "C:/Program Files (x86)/Graphviz2.38/bin/dot.exe",
            viewer = "C:/Program Files (x86)/IrfanView/i_view32.exe")

# Generate image and finally show the Graph 
btg.show_graph(labels = labels)


# Add color gradient for nodes with lower/higher levels of selected nodes
btg.add_color_gradient([6, 16, 22])
btg.show_graph(labels = range(32))