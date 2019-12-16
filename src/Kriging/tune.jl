function tune!(
    krg::Kriging,
    kernbounds,
    meanbounds,
    opt::Optimizer...;
    verbose::Bool = false,
)
    ngp = length(krg.gps)
    for o in opt
        for i in 1:ngp
            fx, x, ret = optimize!(krg.gps[i], kernbounds, meanbounds, o)
            if verbose
                @info "gp[$i] optimized with $(o.algo), fx = $fx"
            end
        end
    end
end

function tune!(
    krg::Kriging,
    kernbounds,
    meanbounds;
    global_eval::Integer = 0,
    global_population::Integer = 0,
    local_eval::Integer = 1000,
    verbose::Bool = false,
)
    if global_eval > 0
        tune!(
            krg,
            kernbounds,
            meanbounds,
            ISRESOptimizer(
                population = global_population,
                maxeval = global_eval,
            ),
            LBFGSOptimizer(maxeval = local_eval),
            verbose = verbose,
        )
    else
        tune!(
            krg,
            kernbounds,
            meanbounds,
            LBFGSOptimizer(maxeval = local_eval),
            verbose = verbose,
        )
    end
end

function optimize!(gp::GP.GPE, kernbounds, meanbounds, opt::Optimizer)
    params_kwargs = GP.get_params_kwargs(
        gp;
        domean = true,
        kern = true,
        noise = false,
        lik = false,
    )
    fobj = if need_gradient(opt)
        (hyp::AbstractVector, grad::AbstractVector) -> begin
            prev = GP.get_params(gp; params_kwargs...)
            try
                GP.set_params!(gp, hyp; params_kwargs...)
                GP.update_target_and_dtarget!(gp; params_kwargs...)
                grad[:] = gp.dtarget
                return gp.target
            catch err
                # reset parameters to remove any NaNs
                GP.set_params!(gp, prev; params_kwargs...)
                if !all(isfinite.(hyp))
                    println(err)
                    return -Inf
                elseif isa(err, ArgumentError)
                    println(err)
                    return -Inf
                elseif isa(err, LinearAlgebra.PosDefException)
                    println(err)
                    return -Inf
                else
                    throw(err)
                end
            end
        end
    else
        (hyp::AbstractVector, grad::AbstractVector) -> begin
            prev = GP.get_params(gp; params_kwargs...)
            try
                GP.set_params!(gp, hyp; params_kwargs...)
                GP.update_target!(gp)
                return gp.target
            catch err
                # reset parameters to remove any NaNs
                GP.set_params!(gp, prev; params_kwargs...)

                if !all(isfinite.(hyp))
                    println(err)
                    return -Inf
                elseif isa(err, ArgumentError)
                    println(err)
                    return -Inf
                elseif isa(err, LinearAlgebra.PosDefException)
                    println(err)
                    return -Inf
                else
                    throw(err)
                end
            end
        end
    end
    init = GP.get_params(gp; params_kwargs...)
    lb, ub = GP.bounds(
        gp,
        nothing,
        meanbounds,
        kernbounds,
        nothing;
        domean = true,
        kern = true,
        noise = false,
        lik = false,
    )
    fx, x, ret = maximize(opt, fobj, lb, ub, init)
    GP.set_params!(gp, x; params_kwargs...)
    GP.update_target_and_dtarget!(gp; params_kwargs...)
    fx, x, ret
end
