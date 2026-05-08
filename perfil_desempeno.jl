
# Cada metodo debe retornar:
# 	Estado (convegio, maximas iteraciones alcanzadas, no convergio)
# 	Numero de iteraciones realizadas
# 	Valor final de ||grad(f)||

function gradiente_const()
end

function gradiente_armijo()
end

function newton_raphson()
end

function gradiente_conj_no_lineal()
end

function bfgs()
end

function dolan_more()
	# construir perfil de desempeno de dolan & more para la metrica de iteraciones.
	# graficar y realizar un breve analisis.
	# sug: el primer paso consiste en ejecutar cada uno de los algoritmos sobre cada 
	# problema y registrr el costo (iteraciones) de cada uno. Asi, podemos formar la 
	# matriz C de costos. Luego se puede calcular una matriz R con la formula de r_sp 
	# (ver clase de perfil de desempeno)
end
