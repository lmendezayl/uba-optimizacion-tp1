# FUNCIONES BENCHMARK
################################################################################
# Para cada función se tiene: f(x), ∇f(x) y Hf(x).
# Los gradientes y Hessianos son ANALÍTICOS para mayor precisión.

# ── F1: Sphere ────────────────────────────────────────────────────────────────
# f(x) = Σ xᵢ²   mínimo en 0, f*=0
sphere(x)      = sum(x .^ 2)
sphere_g(x)    = 2 .* x
sphere_h(x)    = 2 .* Matrix{Float64}(I, length(x), length(x))

# ── F2: Rosenbrock ────────────────────────────────────────────────────────────
# f(x,y) = (1-x)² + 100(y-x²)²   mínimo en (1,1), f*=0
rosenbrock(x)   = (1 - x[1])^2 + 100*(x[2] - x[1]^2)^2
rosenbrock_g(x) = [
    -2*(1 - x[1]) - 400*x[1]*(x[2] - x[1]^2),
     200*(x[2] - x[1]^2)
]
function rosenbrock_h(x)
    h11 = 2 - 400*(x[2] - x[1]^2) + 800*x[1]^2
    h12 = -400*x[1]
    h22 = 200.0
    [h11 h12; h12 h22]
end

# ── F3: Rastrigin (n=2) ───────────────────────────────────────────────────────
# f(x) = 10n + Σ[xᵢ² - 10cos(2πxᵢ)]  mínimo en 0, f*=0
rastrigin(x)   = 10*length(x) + sum(x .^ 2 .- 10 .* cos.(2π .* x))
rastrigin_g(x) = 2 .* x .+ 20π .* sin.(2π .* x)
function rastrigin_h(x)
    d = length(x)
    H = zeros(d, d)
    for i in 1:d
        H[i,i] = 2.0 + 40*π^2 * cos(2π*x[i])
    end
    H
end

# ── F4: Wood (n=4) ────────────────────────────────────────────────────────────
# Función clásica de Fletcher & Powell, generalización de Rosenbrock a 4 variables.
#
# f(x) = 100(x₂ - x₁²)² + (1 - x₁)² + 90(x₄ - x₃²)² + (1 - x₃)²
#       + 10.1[(x₂ - 1)² + (x₄ - 1)²] + 19.8(x₂ - 1)(x₄ - 1)
#
# Mínimo global: x* = (1,1,1,1),  f* = 0
# Características:
#   · No separable: acoplamiento cruzado entre (x₁,x₂) y (x₃,x₄) a través
#     del término 19.8(x₂-1)(x₄-1).
#   · Muy mal condicionada (número de condición de H en x* ≈ 10⁴).
#   · Punto de partida estándar (More, Garbow & Hillstrom 1981):
#     x₀ = (-3, -1, -3, -1).

function wood(x)
    100*(x[2] - x[1]^2)^2 + (1 - x[1])^2 +
    90*(x[4] - x[3]^2)^2  + (1 - x[3])^2 +
    10.1*((x[2] - 1)^2 + (x[4] - 1)^2) +
    19.8*(x[2] - 1)*(x[4] - 1)
end

function wood_g(x)
    g1 = -400*x[1]*(x[2] - x[1]^2) - 2*(1 - x[1])
    g2 =  200*(x[2] - x[1]^2) + 20.2*(x[2] - 1) + 19.8*(x[4] - 1)
    g3 = -360*x[3]*(x[4] - x[3]^2) - 2*(1 - x[3])
    g4 =  180*(x[4] - x[3]^2) + 20.2*(x[4] - 1) + 19.8*(x[2] - 1)
    [g1, g2, g3, g4]
end

function wood_h(x)
    h = zeros(4, 4)
    # ∂²/∂x₁²
    h[1,1] = -400*(x[2] - x[1]^2) + 800*x[1]^2 + 2
    # ∂²/∂x₁∂x₂
    h[1,2] = -400*x[1];  h[2,1] = h[1,2]
    # ∂²/∂x₂²
    h[2,2] = 200 + 20.2
    # ∂²/∂x₃²
    h[3,3] = -360*(x[4] - x[3]^2) + 720*x[3]^2 + 2
    # ∂²/∂x₃∂x₄
    h[3,4] = -360*x[3];  h[4,3] = h[3,4]
    # ∂²/∂x₄²
    h[4,4] = 180 + 20.2
    # ∂²/∂x₂∂x₄ (acoplamiento cruzado)
    h[2,4] = 19.8;  h[4,2] = 19.8
    h
end

# ── F5: Freudenstein & Roth (n=2) ─────────────────────────────────────────────
# Propuesta por Freudenstein & Roth (1963); problema de sistemas no lineales
# reformulado como minimización de suma de cuadrados.
#
# f(x) = (-13 + x₁ + ((5 - x₂)x₂ - 2)x₂)²
#        + (-29 + x₁ + ((x₂ + 1)x₂ - 14)x₂)²
#
# Mínimo global: x* = (5, 4),  f* = 0
# Mínimo local:  x* ≈ (11.41, -0.8968),  f* ≈ 48.98
#
# Características:
#   · La función tiene DOS mínimos locales; dependiendo del punto inicial,
#     los métodos de primer orden pueden quedar atrapados en el local.
#   · Los gradientes tienen magnitud similar cerca de ambos mínimos, lo que
#     dificulta la discriminación por criterio de parada en ‖∇f‖.
#   · Punto de partida estándar: x₀ = (0.5, -2.0).
#   · Es un excelente test para comparar la capacidad de los métodos de
#     escapar mínimos locales y la sensibilidad a la elección de x₀.

function froth(x)
    r1 = -13 + x[1] + ((5 - x[2])*x[2] - 2)*x[2]
    r2 = -29 + x[1] + ((x[2] + 1)*x[2] - 14)*x[2]
    r1^2 + r2^2
end

function froth_g(x)
    r1 = -13 + x[1] + ((5 - x[2])*x[2] - 2)*x[2]
    r2 = -29 + x[1] + ((x[2] + 1)*x[2] - 14)*x[2]
    dr1dx2 = (10*x[2] - 3*x[2]^2 - 2)   # ∂r1/∂x₂
    dr2dx2 = (3*x[2]^2 + 2*x[2] - 14)   # ∂r2/∂x₂
    g1 = 2*r1 + 2*r2                     # ∂f/∂x₁
    g2 = 2*r1*dr1dx2 + 2*r2*dr2dx2      # ∂f/∂x₂
    [g1, g2]
end

function froth_h(x)
    r1 = -13 + x[1] + ((5 - x[2])*x[2] - 2)*x[2]
    r2 = -29 + x[1] + ((x[2] + 1)*x[2] - 14)*x[2]
    dr1dx2 = 10*x[2] - 3*x[2]^2 - 2
    dr2dx2 = 3*x[2]^2 + 2*x[2] - 14
    d2r1dx2 = 10 - 6*x[2]
    d2r2dx2 = 6*x[2] + 2
    h11 = 2 + 2                                              # ∂²f/∂x₁²
    h12 = 2*dr1dx2 + 2*dr2dx2                               # ∂²f/∂x₁∂x₂
    h22 = 2*dr1dx2^2 + 2*r1*d2r1dx2 + 2*dr2dx2^2 + 2*r2*d2r2dx2
    [h11 h12; h12 h22]
end
