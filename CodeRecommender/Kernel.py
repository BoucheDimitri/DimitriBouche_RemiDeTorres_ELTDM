"""
This is the Kernel file which only contains the definition of Kernel class
"""


import pycuda.autoinit
from pycuda.compiler import SourceModule


class Kernel:

	def __init__(self, 
				 name, 
				 code, 
				 const_params={}):
		"""
		Args :
			name (str): the name of the kernel
			code (str): the C code as a string
			const_params (dic): the constants that need to be passed to kernel
		"""
		self.name = name
		self.code = code
		self.constants = const_params
		self.function = None

	def compile(self):
		"""
		Compile kernel using pycuda.compile.SourceModule class
		compiled kernel as a python function is then stored 
		in the function attribute of the Kernel object
		"""
		code = self.code % self.constants
		mod = SourceModule(code)
		self.function = mod.get_function(self.name)