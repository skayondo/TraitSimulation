
# Alternative Model Specification

TraitSimulation provides a variety of model specifications for a user-friendly interface convenient to the users problem setting.

In this notebook we demonstrate the different ways users can specify from a Variance Component Model. Note that the alternative model specification can be applied to the other simulation model in a similar fashion. 

We encourage users to choose whichever specification method best suits their needs. 


```julia
using TraitSimulation, LinearAlgebra
using Random, DataFrames, Test, SnpArrays
Random.seed!(1234)

# First generate some demo-data

n = 10 # number of people
p = 2  # number of fixed effects
d = 2  # number of traits
m = 2  # number of variance components

function generateSPDmatrix(n)
    A = rand(n)
    m = 0.5 * (A * A')
    PDmat = m + (n * Diagonal(ones(n)))
end


function generateRandomVCM(n::Int64, p::Int64, d::Int64, m::Int64)
    # n-by-p design matrix
    X = randn(n, p)
    
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
    return(X, Σ, V)
end

X, Σ, V = generateRandomVCM(n, p, d, m) # generate random data, where variance components are Positive Semi-Definite
B = [[1.0;  0.950498]  [1.0;  0.96467]] # matrix of regression coefficients for non-genetic predictors 
maf = [0.2] # minor allele frequency of simulated SNP
G = snparray_simulation(maf, n) # simulate SnpArray (user can read in their own snps too)
locus = convert(Matrix{Float64}, G, model = ADDITIVE_MODEL, center=false, scale=false); # (convert the snp to Float64)
n_snps = length(maf) 
γ = [0.372575 0.150508]        # matrix of regression coefficients for genetic predictors
variance_formula = @vc Σ[1] ⊗ V[1] + Σ[2] ⊗ V[2] # variance components for model
```




    2-element Array{VarianceComponent,1}:
     Variance Component
      * number of traits: 2
      * sample size: 10
     Variance Component
      * number of traits: 2
      * sample size: 10



## method 1: VCMTrait(X, B, variance_formula)

    **X: Matrix of both genetic and non-genetic predictors**
    **B: Matrix of both genetic and non-genetic regression coefficients**
    **variance_formula: formula to specify the models variance components**

Users who only have a few number of genetic predictors may find this model specification most user friendly. The genetic predictors are concatenated with the non-genetic predictors to form the overall design matrix. Similarly, the regression coefficients of both genetic and non-genetic predictors are concatenated to make the full matrix of regression coefficients.


```julia
X_full = hcat(X, locus)  # full design matrix with both non-genetic and genetic predictors
B_full = vcat(B, γ)      # full matrix of regression coefficients for both non-genetic and genetic predictors 
genetic_model1 = VCMTrait(X_full, B_full, variance_formula)
```




    Variance Component Model
      * number of traits: 2
      * number of variance components: 2
      * sample size: 10




```julia
Random.seed!(1234)
y1 = simulate(genetic_model1)
```




    10×2 Array{Float64,2}:
      6.93768    -3.72705
     -7.26476    -0.0815147
     -0.915509    0.478916
     -5.31707     8.93169
     -0.385205   -3.06236
     13.4914      4.6258
      3.12911     0.639182
      0.178557    1.10218
      2.59476     0.0360991
     -5.39579   -11.2104



## method 2: VCMTrait(mean_ formulas, DataFrame, variance_ formula)

    **mean_formulas: Vector of strings, specifying mean effect for each trait**
    **DataFrame: Named DataFrame of both genetic and non-genetic regression coefficients**
    **variance_formula: formula to specify the models variance components**
    
Users who feel more comfortable using the formulas to specify both the mean and variance of the simulation model may do so on a data frame with names corresponding to the predictors specified in the mean_formula. 


```julia
mean_formulas = ["1.0*predictor1 + 0.950498*predictor2 + 0.372575*locus", "1.0*predictor1 + 0.96467*predictor2 + 0.150508*locus"]
data_frame_X_full = DataFrame(predictor1 = X_full[:, 1], predictor2 = X_full[:, 2], locus = X_full[:, 3])
genetic_model2 = VCMTrait(mean_formulas, data_frame_X_full, variance_formula)
```




    Variance Component Model
      * number of traits: 2
      * number of variance components: 2
      * sample size: 10




```julia
Random.seed!(1234)
y2 = simulate(genetic_model2)
```




    10×2 Array{Float64,2}:
      6.93768    -3.72705
     -7.26476    -0.0815147
     -0.915509    0.478916
     -5.31707     8.93169
     -0.385205   -3.06236
     13.4914      4.6258
      3.12911     0.639182
      0.178557    1.10218
      2.59476     0.0360991
     -5.39579   -11.2104



## method 3: VCMTrait(X, B, Σ, V)

To specify many variance components, users can decide not to use formulas and specify just the necessary mean and variance components as follows.

    **X_full: Matrix of both genetic and non-genetic predictors**
    **B_full: Matrix of both genetic and non-genetic regression coefficients**
    **Σ: Collection of all m Variance Components **
    **V: Collection of all m Variance Covariance Matrices**
    
For users who have potentially many non-genetic predictors and many variance components, the formula specification can be avoided alltogether.


```julia
Σ = [Σ...] # Σ = [Σ[1], Σ[2]] collect all the variance components 
V = [V...] # V = [V[1], V[2]] collect all the variance covariance matrices
genetic_model3 = VCMTrait(X_full, B_full, Σ, V)
```




    Variance Component Model
      * number of traits: 2
      * number of variance components: 2
      * sample size: 10




```julia
Random.seed!(1234)
y3 = simulate(genetic_model3)
```




    10×2 Array{Float64,2}:
      6.93768    -3.72705
     -7.26476    -0.0815147
     -0.915509    0.478916
     -5.31707     8.93169
     -0.385205   -3.06236
     13.4914      4.6258
      3.12911     0.639182
      0.178557    1.10218
      2.59476     0.0360991
     -5.39579   -11.2104



## method 4: VCMTrait(X, B, G, γ, variance_formula)

    **X: Matrix of non-genetic predictors**
    **B: Matrix of regression coefficients for non-genetic predictors**
    **G: Matrix of genetic predictors (specified as a SnpArray)**
    **γ: Matrix of regression coefficients for genetic predictors (snps)**
    **variance_formula: formula to specify the models variance components**
    
    
For users who have potentially many non-genetic predictors and a vast number of SNPs, the SNPs may be specified separately as a SnpArray for maximum efficiency. For users who are interested in this case, please take a look at the SnpArrays documentation for more details on [LinearAlgebra using SnpArrays](https://openmendel.github.io/SnpArrays.jl/latest/#Linear-algebra-1)



```julia
genetic_model4 =  VCMTrait(X, B, G, γ, variance_formula)
```




    Variance Component Model
      * number of traits: 2
      * number of variance components: 2
      * sample size: 10




```julia
Random.seed!(1234)
y4 = simulate(genetic_model4)
```




    10×2 Array{Float64,2}:
      6.93768    -3.72705
     -7.26476    -0.0815147
     -0.915509    0.478916
     -5.31707     8.93169
     -0.385205   -3.06236
     13.4914      4.6258
      3.12911     0.639182
      0.178557    1.10218
      2.59476     0.0360991
     -5.39579   -11.2104



## method 5: VCMTrait(X, B, G, γ,  Σ, V)


    **X: Matrix of non-genetic predictors**
    **B: Matrix of regression coefficients for non-genetic predictors**
    **G: Matrix of genetic predictors (specified as a SnpArray)**
    **γ: Matrix of regression coefficients for genetic predictors (snps)**
    **Σ: Collection of all m Variance Components **
    **V: Collection of all m Variance Covariance Matrices**
    
For users who have potentially many non-genetic predictors, a vast number of SNPs, and many variance components, the SNPs may be specified separately as a SnpArray for maximum efficiency and the variance components may be specified as a list. For users who are interested in this case, please take a look at the SnpArrays documentation for more details on [LinearAlgebra using SnpArrays](https://openmendel.github.io/SnpArrays.jl/latest/#Linear-algebra-1)


```julia
genetic_model5 =  VCMTrait(X, B, G, γ, [Σ...], [V...])
```




    Variance Component Model
      * number of traits: 2
      * number of variance components: 2
      * sample size: 10




```julia
Random.seed!(1234)
y5 = simulate(genetic_model5)
```




    10×2 Array{Float64,2}:
      6.93768    -3.72705
     -7.26476    -0.0815147
     -0.915509    0.478916
     -5.31707     8.93169
     -0.385205   -3.06236
     13.4914      4.6258
      3.12911     0.639182
      0.178557    1.10218
      2.59476     0.0360991
     -5.39579   -11.2104


