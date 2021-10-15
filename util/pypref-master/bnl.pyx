

from libcpp.list cimport list as stdlist
from libcpp.vector cimport vector as stdvector
from libcpp cimport bool

from cython.operator cimport dereference as deref, preincrement as inc

import array

# We need both the standard python import and the additional c-files import
cimport numpy as np
import numpy as np

cimport cython

cdef class cpref:

  cdef public int next_id     
    
  # Abstract compare/equal methods, to be overwritten
  cdef bool cmp(self, int t1, int t2): return False
  cdef bool eq( self, int t1, int t2): return False  
    

# Base / score preference
cdef class cbasepref(cpref):

  cdef double[:] arr
  
  def __init__(self, int next_id_, double[:] arr): 
    self.next_id = next_id_ + 1
    self.arr = arr
      
  cdef bool cmp(self, int t1, int t2):
    return self.arr[t1] < self.arr[t2]

  cdef bool eq(self, int t1, int t2):
    return self.arr[t1] == self.arr[t2]
    
    
cdef class creversepref(cpref):

  cdef public cpref p
  
  cdef bool eq(self, int t1, int t2):
    return self.p.eq(t1, t2)
    
  cdef bool cmp(self, int t1, int t2): # swap t1 and t2
    return self.p.cmp(t2, t1)
    
    
cdef class ccomplexpref(cpref):

  cdef public cpref p1
  cdef public cpref p2
  
  cdef bool eq(self, int t1, int t2):
    return self.p1.eq(t1, t2) and self.p2.eq(t1, t2)
    
    
cdef class cparetopref(ccomplexpref):
    
  cdef bool cmp(self, int t1, int t2):
    return ( ((self.p1.cmp(t1, t2) or self.p1.eq(t1, t2)) and self.p2.cmp(t1, t2)) or 
             ((self.p2.cmp(t1, t2) or self.p2.eq(t1, t2)) and self.p1.cmp(t1, t2))    )


cdef class cpriorpref(ccomplexpref):
   
  cdef bool cmp(self, int t1, int t2):
    return self.p1.cmp(t1, t2) or (self.p1.eq(t1, t2) and self.p2.cmp(t1, t2))
    
    
cdef class cintersectpref(ccomplexpref):
   
  cdef bool cmp(self, int t1, int t2):
    return self.p1.cmp(t1, t2) and self.p2.cmp(t1, t2)
        
        
cdef class cunionpref(ccomplexpref):
   
  cdef bool cmp(self, int t1, int t2):
    return self.p1.cmp(t1, t2) or self.p2.cmp(t1, t2)
        

cdef cpref deserialize_pref(list lst, np.ndarray[double, mode="c", ndim=2] arrs, int next_id):
    
  cdef int p_char = <int>(lst[0])
  cdef cpref p1
  cdef cpref p2
  cdef cpref res_pref

  if p_char in [38, 42, 43, 124]: # Binary complex preference
  
    if (len(lst) != 3): raise ValueError("Unexpected length in serialized preference list")
      
    if   p_char == 42:  res_pref = cparetopref()    # ord('*')
    elif p_char == 38:  res_pref = cpriorpref()     # ord('&')
    elif p_char == 124: res_pref = cintersectpref() # ord('|')
    elif p_char == 43:  res_pref = cunionpref()     # ord('+')
  
    res_pref.p1 = deserialize_pref(<list>(lst[1]), arrs, next_id)
    res_pref.p2 = deserialize_pref(<list>(lst[2]), arrs, res_pref.p1.next_id)
    res_pref.next_id = res_pref.p2.next_id
    
    return res_pref
    
  elif p_char == 45: # ord('-') ==> Reverse preference
  
    if (len(lst) != 2): raise ValueError("Unexpected length in serialized preference list")  
  
    res_pref = creversepref()
    res_pref.p = deserialize_pref(<list>(lst[1]), arrs, next_id)
    res_pref.next_id = res_pref.p.next_id
  
    return res_pref
  
  elif p_char == 115: # ord('s') ==> Base (score) preference
  
    return cbasepref(next_id, arrs[next_id,:])
    
  else:
    raise ValueError("Unexpected symbol in serialized preference")

# ------------------------------------------------------------------------------------------------------


# stdlist is automatically converted to a python list; boundscheck=False slightly increases performance
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef stdlist[int] runbnl(np.ndarray[double, mode="c", ndim=2] arrs, list serialized_pref):  
  
  cdef cpref p = deserialize_pref(serialized_pref, arrs, 0)
  
  cdef stdlist[int] window
  cdef stdlist[int].iterator j
  cdef bool dominated
  
  cdef int ntuples = <int>(len(arrs[0,:]))
  
  # create index vector
  cdef int[:] v = array.array('i', range(ntuples))
  
  # Add first element to window
  window.push_back(v[0])  
  
  # If i is not typed, the v[i] command raises a performance warning!
  cdef int i
  for i in range(1, ntuples):
    dominated = False
    j = window.begin()
    while(j != window.end()):
      if p.cmp(deref(j), v[i]): # j (window element) is better
        dominated = True
        break
      elif p.cmp(v[i], deref(j)): # arr[i] (picked element) is better
        # erase element and procede with next one
        j = window.erase(j)
        if j == window.end(): break
        else:                 continue
      inc(j) # increment list pointer
    # if not dominated in inner loop, add to window
    if not dominated:
      window.push_back(v[i]) 
     
  return(window)
  
  
# ------------------------------------------------------------------------------------------------------

# TOP-k interface
# ===============

# On step of BNL for top-k selection, changes remainder (array!)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef stdlist[int] bnl_remainder(stdvector[int]& v, stdvector[int]& remainder, cpref p):
  
  cdef bool dominated
  cdef int count = 0
  cdef int ntuples = <int>(v.size())
  cdef stdlist[int].iterator j
  cdef stdlist[int] window
  
  if (ntuples == 0): return(window)
  
  # add first element to window
  window.push_back(v[0])  
  
  cdef int i
  for i in range(1, ntuples):
    dominated = False
    j = window.begin()
    while(j != window.end()):
      if p.cmp(deref(j), v[i]): # j (window element) is better
        # add to remainder
        remainder[count] = v[i]
        count += 1
        # set dominated-flag and break
        dominated = True
        break
      elif p.cmp(v[i], deref(j)): # arr[i] (picked element) is better
        # put element to remainder
        remainder[count] = deref(j)
        count += 1
        # erase element and procede with next one
        j = window.erase(j)
        if j == window.end(): break
        else:                 continue
      inc(j) # increment list pointer
    # if not dominated in inner loop, add to window
    if not dominated:
      window.push_back(v[i]) 
      
  # Cut remainder
  remainder.resize(count)
  
  return(window)


# get reached level and reached tuple count
# topk, at_least and top_level are "-1" if not active
cdef bool do_break(int level, int count, int topk, int at_least, int top_level, bool and_connected):
  
  if and_connected: # (intersection, some break conditions suffices)
    # break condition does not count as a break condition if caused by a "-1"!
    if ((topk     != -1 and count >= topk    ) or
        (at_least != -1 and count >= at_least) or
        (level == top_level)):
          return True
  else: # or-connected (union; all break conditions have to be fullfilled)
    # if the top-value is "-1" they are trivially true!
    if (count >= topk and count >= at_least and level >= top_level):
      return True
        
  return False
  
  


# Returns a list of length 2, containing the indices and the level values
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef stdlist[stdlist[int]] runbnl_top(np.ndarray[double, mode="c", ndim=2] arrs, list serialized_pref, 
                                       int topk, int at_least, int top_level, bool and_connected):  
  
  cdef cpref p = deserialize_pref(serialized_pref, arrs, 0)
  
  cdef stdlist[int] levels # All levels in same order as in return value
  cdef stdlist[int] tuple_indices # All tuple indices
  
  cdef int ntuples = <int>(len(arrs[0,:]))

  # Initial values of indices vector and remainder (arbitrary values)
  cdef stdvector[int] v = stdvector[int](ntuples)
  cdef stdvector[int] remainder = stdvector[int](ntuples)
  cdef int i
  for i in range(ntuples):
    v[i] = i;
    
  # Counter for tuples/levels
  cdef int count = 0
  cdef int level = 0
  
  # Temporary result (one bnl iteration)
  cdef stdlist[int] tmp_res
  cdef int tmp_size
    
  while(not do_break(level, count, topk, at_least, top_level, and_connected)):
    tmp_res = bnl_remainder(v, remainder, p)
    tmp_size = <int>tmp_res.size()
    if tmp_size == 0: break # no more tuples
    count += tmp_size
    # Append temp result to final result
    tuple_indices.splice(tuple_indices.end(), tmp_res)
    level += 1
    for i in range(tmp_size):
      levels.push_back(level)
    # Interchange v and remainder
    v.swap(remainder)
    
  # cut? # avoid resize with negative values by topk >= 0
  if (topk >= 0 and topk < count and (and_connected or (top_level == -1 and at_least == -1))):
    tuple_indices.resize(topk, 0)
    levels.resize(topk, 0)
    
  # Combine indices and level and return
  return [tuple_indices, levels]
  
  
# ------------------------------------------------------------------------------------------------------
  
# Hasse diagram
  
  
# Returns a list of length 2, containing the "from" and the "to" field
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[int, mode="c", ndim=2] get_transitive_reduction(np.ndarray[double, mode="c", ndim=2] arrs, list serialized_pref):
  
  cdef cpref p = deserialize_pref(serialized_pref, arrs, 0)
  
  cdef stdlist[int] edge_from, edge_to
  cdef int ntuples = <int>(len(arrs[0,:]))
  
  cdef int i, j, k
  cdef bool found
  
  for i in range(ntuples):
    for j in range(ntuples):
      if p.cmp(i, j):
        found = False
        for k in range(ntuples):
          if (p.cmp(i, k) and p.cmp(k, j)):
            found = True
        if not found:
          edge_from.push_back(i)
          edge_to.push_back(j)
  
  
  return np.array([edge_from, edge_to], dtype = np.dtype("i"))
