
import NLopt, ForwardDiff, DiffResults

import Base: setproperty!, getproperty
function auto_diff(f)
    (x, g) -> if !isempty(g)
        res = DiffResults.DiffResult(0.0, g)
        ForwardDiff.gradient!(res, f, x)
        res.value
    else
        f(x)
    end
end

function combine_gradient(f, gf)
    (x, g) -> if !isempty(g)
        g .= gf(x)
        f(x)
    else
        f(x)
    end
end

function wrap_function(f, g)
    if g === :Auto || g === :AUTO
        auto_diff(f)
    else
        combine_gradient(f, g)
    end
end


function create_nlopt(algo, lb, ub; options...)
    opt = NLopt.Opt(algo, length(lb))
    opt.lower_bounds = lb
    opt.upper_bounds = ub
    foreach(options) do (p, v)
        setproperty!(opt, p, v)
    end
    return opt
end
