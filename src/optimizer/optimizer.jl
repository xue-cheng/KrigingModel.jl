include("utils.jl")

abstract type Optimizer end

abstract type GlobalOptimizer <: Optimizer end

abstract type LocalOptimizer <: Optimizer end


isglobal(::GlobalOptimizer) = true
isglobal(::LocalOptimizer) = false

minimize(o::Optimizer, f, g, lb, ub, x) =
    minimize(o, wrap_function(f, g), lb, ub, x)
maximize(o::Optimizer, f, g, lb, ub, x) =
    maximize(o, wrap_function(f, g), lb, ub, x)
isglobal(o::Optimizer) = error("not implemented")
need_gradient(o::Optimizer) = error("not implemented")


include("stochastic_opt.jl")
include("gradient_opt.jl")
include("gradfree_opt.jl")
