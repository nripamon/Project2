class Optimiser():
	def __init__(self,eta):
		self.eta = eta

	def step(self,module):
		raise NotImplementedError

class SGD(Optimiser):
	def __init__(self,eta):
		self.eta = eta

	def step(self,module):
		module.update_param(self.eta)


