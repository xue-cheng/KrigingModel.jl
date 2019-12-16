struct GradFreeLocalOptimizer <: LocalOptimizer
    algo::Symbol
    options::Dict{Symbol,Any}
end

need_gradient(::GradFreeLocalOptimizer) = false

function BOBYQAOptimizer(; options...)
    GradFreeLocalOptimizer(:LN_BOBYQA, Dict{Symbol,Any}(options))
end

function minimize(o::GradFreeLocalOptimizer, f, lb, ub, x0)
    opt = create_nlopt(o.algo, lb, ub; min_objective = f, o.options...)
    fx, x, ret = NLopt.optimize(opt, x0)
    ret == :FORCED_STOP && throw(InterruptException())
    return fx, x, ret
end

function maximize(o::GradFreeLocalOptimizer, f, lb, ub, x0)
    opt = create_nlopt(o.algo, lb, ub; max_objective = f, o.options...)
    fx, x, ret = NLopt.optimize(opt, x0)
    ret == :FORCED_STOP && throw(InterruptException())
    return fx, x, ret
end
