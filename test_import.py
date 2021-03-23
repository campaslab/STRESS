import sys

if sys.version_info[0] < 3:
	raise Exception("Must be using Python 3")
else:
	print("Using python3")

try:
	import numpy
	print("module 'numpy' is installed")
except:
	raise Exception("module 'numpy' is not installed")


try:
	import scipy
	print("module 'scipy' is installed")
except:
	raise Exception("module 'scipy' is not installed")

try:
	import gdist
	print("module 'gdist' ('tvb-gdist') is installed")
except:
	raise Exception("module 'gdist' ('tvb-gdist') is not installed")

try:
	import mpmath
	print("module 'mpmath' is installed")
except:
	raise Exception("module 'mpmath' is not installed")

try:
	import matplotlib
	print("module 'matplotlib' is installed")
except:
	raise Exception("module 'matplotlib' is not installed")

try:
	import seaborn
	print("module 'seaborn' is installed")
except:
	raise Exception("module 'seaborn' is not installed")

try:
	import vtk
	print("module 'vtk' is installed")
except:
	raise Exception("module 'vtk' is not installed")
