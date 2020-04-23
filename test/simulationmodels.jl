using Random, SnpArrays, DataFrames, GLM
using LinearAlgebra
using BenchmarkTools
beta = [1.0, 5.0]

n = 10
p = 2
d = 2
m = 2

function generateSPDmatrix(n)
	A = rand(n)
	m = 0.5 * (A * A')
	PDmat = m + (n * Diagonal(ones(n)))
end


function generateRandomVCM(n::Int64, p::Int64, d::Int64, m::Int64)
	# n-by-p design matrix
	X = randn(n, p)

	# p-by-d mean component regression coefficient for each trait
	B = hcat(ones(p, 1), rand(p))

	V = ntuple(x -> zeros(n, n), m)
	for i = 1:m-1
	  copy!(V[i], generateSPDmatrix(n))
	end
	copy!(V[end], Diagonal(ones(n))) # last covarianec matrix is identity

	# a tuple of m d-by-d variance component parameters
	Σ = ntuple(x -> zeros(d, d), m)
	for i in 1:m
	  copy!(Σ[i], generateSPDmatrix(d))
	end
	return(X, B, Σ, V)
end

X, B, Σ, V = generateRandomVCM(n, p, d, m)

glmtraitobject = GLMTrait(Matrix(df), beta, dist, link)
y_normal  = simulate(glmtraitobject)
@test size(y_normal) == size(Matrix(df)*beta)

number_independent_simulations  = 5
@test size(simulate(glmtraitobject, number_independent_simulations)) == (n, number_independent_simulations)

@test glmtraitobject isa TraitSimulation.AbstractTraitModel

@test GLMTrait(Matrix(df), beta, dist, link).η == evaluated_output[1]

# make sure that ordered multinomial models works
link = LogitLink()
θ = [1.0, 1.2, 1.4]
Ordinal_Model_Test = OrderedMultinomialTrait(X, beta, θ, link)

@test Ordinal_Model_Test isa TraitSimulation.AbstractTraitModel
@test eltype(simulate(Ordinal_Model_Test)) == Int64
@test size(simulate(Ordinal_Model_Test, number_independent_simulations))  == (n, number_independent_simulations)

@test noutcomecategories(Ordinal_Model_Test) == 4

# make sure GLM simulation works
dist = Poisson()
link = IdentityLink()
beta = [21.0, 5.0]
glmtraitobject2 = GLMTrait(abs.(X), beta, dist, link)

# check if clamper worked
@test sum(glmtraitobject2.η .> 20.0) == 0

@test  eltype(simulate(glmtraitobject2)) == Int64

dist = Bernoulli()
link = LogitLink()
glmtraitobject3 = GLMTrait(X, beta, dist, link)

@test eltype(simulate(glmtraitobject3)) == Bool

@test size(simulate(glmtraitobject3, number_independent_simulations))  == (n, number_independent_simulations)

dist = NegativeBinomial()
link = LogLink()
glmtraitobject4 = GLMTrait(X, beta, dist, link)

@test eltype(simulate(glmtraitobject4)) == Int64

@test size(simulate(glmtraitobject4, number_independent_simulations))  == (n, number_independent_simulations)

dist = Gamma()
link = LogLink()
glmtraitobject5 = GLMTrait(X, beta, dist, link)

@test eltype(simulate(glmtraitobject5)) == Float64

@test size(simulate(glmtraitobject5, number_independent_simulations))  == (n, number_independent_simulations)

# make sure GLMM works
glmtraitobject6 = GLMMTrait(X, B, varcomp, dist, link)
@test glmtraitobject6 isa TraitSimulation.AbstractTraitModel

@test nsamplesize(glmtraitobject6) == n
@test neffects(glmtraitobject6) == p
@test nvc(glmtraitobject6) == m
@test ntraits(glmtraitobject6) == d


y_glmm = simulate(glmtraitobject6)
@test size(y_glmm) == (size(X,  1),  size(B, 2))

@test length(simulate(glmtraitobject6, number_independent_simulations)) == number_independent_simulations


# simulate the SnpArray
G = snparray_simulation([0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5], n)
γ = rand(7)

glmtraitobject7 =  GLMTrait(X, beta, G, γ, dist, link)

@test glmtraitobject7.dist  == Gamma
@test eltype(simulate(glmtraitobject7)) == eltype(dist)

@test size(simulate(glmtraitobject7, number_independent_simulations), 2) == number_independent_simulations

@test GLMTrait(X, beta, G, γ, dist, link) != empty

@test nsamplesize(glmtraitobject7) == n
@test neffects(glmtraitobject7) == p
