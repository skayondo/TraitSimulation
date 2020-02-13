using DataFrames
using LinearAlgebra
using TraitSimulation
using Random

"""
VarianceComponent
this VarianceComponent type stores A, B , CholA and CholB so we don't have to compute the cholesky decomposition inside the loop.
"""
struct VarianceComponent
	A::Matrix{Float64} # n_traits by n_traits
	B::Matrix{Float64} # n_people by n_people
	CholA::Cholesky{Float64,Array{Float64,2}} # cholesky decomposition of A
	CholB::Cholesky{Float64,Array{Float64,2}} # cholesky decomposition of B
	function VarianceComponent(A, B) #inner constructor given A, B
		return new(A, B, cholesky(Symmetric(A)), cholesky(Symmetric(B))) # stores these values (this is helpful so we don't have it inside the loop)
	end
end


"""
LMM_trait_simulation
single LMM trait with given evaluated covariance matrix so not a VarianceComponent type
 i.e given an evaluated matrix of means, simulate from LMM.
"""
function LMM_trait_simulation(mu, vc::AbstractArray{T, 2}) where T
	n_people = size(mu)[1]
	n_traits = size(mu)[2]
	simulated_trait = zeros(n_people, n_traits)
	z = Matrix{Float64}(undef, n_people, n_traits)
	chol_Σ = cholesky(Symmetric(vc)) #for a single evaluated matrix as the specified covariance matrix
	randn!(z) #generate from standard normal
	lmul!(chol_Σ.L, z)
	simulated_trait += z
	simulated_trait += mu
	return simulated_trait
end

"""
append_terms
Allows us to append terms to create a VarianceComponent type
"""
function append_terms!(AB, summand)
	A_esc = esc(summand.args[2])	# elements in args are symbols,
	B_esc = esc(summand.args[3])
	push!(AB.args, :(VarianceComponent($A_esc, $B_esc)))
end

"""
this vc macro allows us to create a vector of VarianceComponent objects for simulation so with_bigfloat_precis, precision::Integer)
so that the user can type out @vc V[1] ⊗ Σ[1] + V[2] ⊗ Σ[2] + .... + V[m] ⊗ Σ[m]
"""
macro vc(expression)
	n = length(expression.args)
	AB = :(VarianceComponent[]) # AB is an empty vector of variance components list of symbols
	if expression.args[1] != :+ #if first argument is not plus (only one vc)
		summand = expression
		append_terms!(AB, summand)
	else #MULTIPLE VARIANCE COMPONENTS if the first argument is a plus (Sigma is a sum multiple variance components)
		for i in 2:n
			summand = expression.args[i]
			append_terms!(AB, summand)
		end
	end
	return(:($AB))
end

"""
vcobjectuple(vcobject)
This function creates a tuple of Variance Components, given a vector of variancecomponent objects to be compatible with VarianceComponentModels.jl
"""
function  vcobjtuple(vcobject::Vector{VarianceComponent})
	m = length(vcobject)
	d = size(vcobject[1].A, 1)
	n = size(vcobject[1].B, 1)
	Σ = ntuple(x -> zeros(d, d), m)
	V = ntuple(x -> zeros(n, n), m)
	for i in eachindex(vcobject)
		copyto!(V[i], vcobject[i].B)
		copyto!(Σ[i], vcobject[i].A)
	end
	return(Σ, V)
end

"""
SimulateMVN!(z, vc)
SimulateMVN(n_people, n_traits, vc::VarianceComponent)
For a single Variance Component, algorithm that will transform standard normal distribution.
SimulateMVN allows us to preallocate n_people by n_traits and rewrite over this matrix to save memory allocation.
"""
function SimulateMN!(Z::Matrix, vc::VarianceComponent)
	cholA = vc.CholA # grab (not calculate) the stored Cholesky decomposition of n_traits by n_traits variance component matrix
	cholB = vc.CholB # grab (not calculate) the stored Cholesky decomposition of n_people by n_people variance component matrix
	randn!(Z)
	lmul!(cholB.L, Z) # Z => (CholB.L)Z
	rmul!(Z, cholA.U) # Z => (CholB.L)Z(CholA.U) so each Y_i = Z ~ MN(0, A_i = (CholB.L)(CholB.L)^T, B_i = (CholA.U)^T(CholA.U)), i in 1:m
	return(Z) #adds onto Z the effects of each variance component
end

function SimulateMN(n_people, n_traits, vc::VarianceComponent)
	Z = Matrix{Float64}(undef, n_people, n_traits)
	SimulateMN!(Z, vc::VarianceComponent) # calls the function to apply the cholesky decomposition (we have stored in vc object)and transforms/updates z ~ MN(0, vc)
	return(Z) #returns the allocated and now transformed Z
end

"""
LMM_trait_simulation(mu, vc::VarianceComponent)
For a single Variance Component object, without computing mean from dataframe and formulas
"""
function LMM_trait_simulation(mu, vc::VarianceComponent)
	n_people = size(mu)[1]
	n_traits = size(mu)[2]
	Z = SimulateMN(n_people, n_traits, vc)
	Z += mu
	return Z
end

"""
Aggregate_VarianceComponents(z, total_variance, vc)
Update the simulated trait with the effect of each variance component. We note the exclamation is to indicate this function will mutate or override the values that its given.
"""
function Aggregate_VarianceComponents!(Z::Matrix, total_variance, vc::Vector{VarianceComponent})
	for i in 1:length(vc)
		SimulateMN!(Z, vc[i]) # this returns LZUt -> vec(LZUt) ~ MVN(0, Σ_i ⊗ V_i)
		total_variance .+= Z #add the effects of each variance component
	end
	return total_variance
end

"""
LMM_trait_simulation(mu, vc::Vector{VarianceComponent})
For a vector of Variance Component objects, without computing mean from dataframe and formulas i.e given an evaluated matrix of means, simulate from LMM.
"""
function LMM_trait_simulation(mu::Matrix, vc::Vector{VarianceComponent})
	n_people = size(mu)[1]
	n_traits = size(mu)[2]
	simulated_trait = zeros(n_people, n_traits) #preallocate memory for MVN np x 1 vector later to be reshaped into matrix n x p
	Z = Matrix{Float64}(undef, n_people, n_traits)
	Aggregate_VarianceComponents!(Z, simulated_trait, vc) # sum up the m independent, np x 1 vectors, Y = sum( Yi ~ MVN(0, A_i ⊗ B_i) , i in 1:m)
	simulated_trait += mu # add the mean matrix
	return simulated_trait
end

#
function LMM_trait_simulation(X::AbstractArray{T, 2}, B::Matrix{Float64}, Σ, V) where T
	n, p = size(X)
	m = length(V)
	d = size(Σ, 1)
	VC = [VarianceComponent(Σ[i], V[i]) for i in 1:length(V)]
	mean = X*B
	VCM_Model = LMMTrait(mean, VC)
	VCM_trait = simulate(VCM_Model)
	return VCM_trait
end

function LMM_trait_simulation(X::AbstractArray{T, 2}, B::AbstractArray{Float64}, VC::Vector{VarianceComponent}) where T
	mean = X*B
	VCM_Model = LMMTrait(mean, VC)
	VCM_trait = simulate(VCM_Model)
	return VCM_trait
end