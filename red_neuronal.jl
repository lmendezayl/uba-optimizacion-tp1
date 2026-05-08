using LinearAlgebra, RDatasets, Random, Statistics, Plots

iris = dataset("datasets", "iris")
X = Matrix(iris[:, 1:4])
y = iris.Species

mutable struct RedNeuronal{T<:Real}
	# Utilizaremos una red neuronal con tres capas con funciones de activacion ReLU,
	# identidad y softmax de 5 niveles.
	#
	# Nota: no es como Python, no hace falta definir una clase, ni tapoco
	# hace falta definir atributos como __init__ para poder definir el llamado a 
	# el constructor de esta clase. Aca directamente llamamos 
	# m = RedNeuronal(W1, b1, W2, b2) 
	# y listo, con esto es suficiente para crear el modelo.
	#
	# El downside es que no podemos crear metodos bajo el struct RedNeuronal
	W1::Matrix{T} # l1 -> l2
	b1::Vector{T}
	W2::Matrix{T} # l2 -> l3
	b2::Vector{T}
end

# creamos el modelo con el constructor de Red Neuronal
Random.seed!(73)
W1 = randn(5, size(X_train, 2))
b1 = randn(5)
W2 = randn(size(y_train, 1), 5)
b2 = randn(size(y_train, 1 ))

model = RedNeuronal(W1, b1, W2, b2)

# funciones de activacion
relu(x) = max.(0,x)
id(x) = x
softmax(x) = exp.(x) / sum(exp.(x))

function forward_pass(model, x)
	# Implementar una funcion que para cada dato x retorne la prediccion segun 
	# el modelo.
	W1, b1, W2, b2 = model.W1, model.b1, model.W2, model.b2
	z_1 = W1 * x + b1
	a_1 = relu(z_1)
	z_2 = W2 * a_1 + b2
	a_2 = id(z_2) # redundante pero descriptivo
	y = softmax(a_2)
	return y	
end

# TODO
function grad(model, x, y)
	# Implementar una funcion que calcule el gradiente de la funcion de perdida para 
	# cada par de datos (x,y) utilizando backprop. Usarla para calcular el grad completo
	# usando la funcion grad_total de utils_tools.jl
		
end

# TODO
function train!(model, train_set, step, max_iter)
	# La funcion debe retornar el modelo entrenado y un vector ocn los valores de la 
	# funcion de perdida en cada iteracion
	# Opcional: Probar distintas funciones de activacion y comparar
	cross_entropy(y, y_hat) = -sum(y*log.(y_hat))
	for i in 1:max_iter
		# begin iter
		# correr forward pass -> y
		# calcular cross entropy -> loss 
		# hacer backprop -> delta_C
		# actualizar? preguntar
		# end iter	
	end
end

# TODO
function results()
	# Implementar una funcion que retorne un grafico de los valores de la funcion 
	# de perdida para cada iteracion y el rendimiento de nuestro modelo tanto en
	# los conjuntos de entrenamiento como en los de prueba, por ejemplo calculando
	#
	# 				 # favorables
	# 		       		---------------
	# 		      		# casos totales

end



