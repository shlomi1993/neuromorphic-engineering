from nengo import Network, Node, Ensemble, Connection, Vector, PES

model = Network()
with model:
	stim = Node([0, 0])
	a = Ensemble(n_neurons=100, dimensions=1)
	Connection(stim, a)

	b = Ensemble(n_neurons=50, dimensions=1)

	def custom_func(x: Vector) -> float:
		return 0

	c = Connection(a, b, function=custom_func, learning_rule=PES())

	error = Ensemble(n_neurons=100, dimensions=1)
	Connection(b, error)
	Connection(error, c.learning_rule)

	def target_func(x: Vector) -> float:
		return x ** 2

	Connection(stim, error, function=target_func, transform=-1)

