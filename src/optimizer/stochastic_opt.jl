
struct StochasticSearchOptimizer{P,S} <: GlobalOptimizer
    algo::Symbol
    population::P
    srand::S
    options::Dict{Symbol,Any}
end

need_gradient(::StochasticSearchOptimizer) = false



function CRSOptimizer(; population = 0, srand = nothing, options...)
    StochasticSearchOptimizer(
        :GN_CRS2_LM,
        population,
        srand,
        Dict{Symbol,Any}(options),
    )
end

function ISRESOptimizer(; population = 0, srand = nothing, options...)
    StochasticSearchOptimizer(
        :GN_ISRES,
        population,
        srand,
        Dict{Symbol,Any}(options),
    )
end


function calc_population(o::StochasticSearchOptimizer, lb, ub)
    if isa(o.population, Real)
        convert(Int, o.population)
    else
        convert(Int, o.population(lb, ub))
    end
end

function minimize(o::StochasticSearchOptimizer, f, lb, ub, x0)
    opt = create_nlopt(o.algo, lb, ub; min_objective = f, o.options...)
    pop = calc_population(o, lb, ub)
    !isnothing(pop) && NLopt.population!(opt, pop)
    !isnothing(o.srand) && NLopt.srand(o.srand)
    fx, x, ret = NLopt.optimize(opt, x0)
    NLopt.srand_time()
    ret == :FORCED_STOP && throw(InterruptException())
    return fx, x, ret
end

function maximize(o::StochasticSearchOptimizer, f, lb, ub, x0)
    opt = create_nlopt(o.algo, lb, ub; max_objective = f, o.options...)
    pop = calc_population(o, lb, ub)
    !isnothing(pop) && NLopt.population!(opt, pop)
    !isnothing(o.srand) && NLopt.srand(o.srand)
    fx, x, ret = NLopt.optimize(opt, x0)
    NLopt.srand_time()
    ret == :FORCED_STOP && throw(InterruptException())
    return fx, x, ret
end
