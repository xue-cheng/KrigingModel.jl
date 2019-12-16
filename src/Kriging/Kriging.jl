mutable struct Kriging{N,M,XS<:Scaler,YS<:Scaler}
    xscaler::XS
    yscaler::YS
    gps::Vector{GP.GPE}
end

include("tune.jl")
include("predict.jl")
include("append.jl")
function Kriging(
    x::AbstractMatrix,
    y::AbstractMatrix,
    mean::GP.Mean,
    kernel::GP.Kernel;
    xscaler::Scaler = StandardScaler(x),
    yscaler::Scaler = StandardScaler(y),
    capacity::Integer = 1024,
    stepsize::Integer = 1024,
)
    nx, mx = size(x)
    ny, my = size(y)
    if mx != my
        throw(ArgumentError("`x` and `y` must have same number of observations"))
    end
    xs = xscaler * x
    ys = yscaler * y
    logNoise = log((1000 + mx) * eps())
    if isa(mean, GP.Mean)
        mean = [deepcopy(mean) for i in 1:ny]
    elseif length(mean) != ny || !all(isa(mean, GP.Mean))
        throw(ArgumentError("`mean` must be a GP.Mean object OR a list/tuple of $ny GP.Mean objects"))
    end

    if isa(kernel, GP.Kernel)
        kernel = [deepcopy(kernel) for i in 1:ny]
    elseif length(kernel) != ny || !all(isa.(kernel, GP.Kernel))
        throw(ArgumentError("`kernel` must be a GP.Kernel object OR a list/tuple of $ny GP.Kernel objects"))
    end
    gps = Vector{GP.GPE}(undef, ny)
    for i in 1:ny
        gps[i] = GP.ElasticGPE(
            nx;
            mean = mean[i],
            kernel = kernel[i],
            logNoise = logNoise,
            capacity = capacity,
            stepsize = stepsize,
        )
        append!(gps[i], xs, ys[i, :])
    end
    Kriging{nx,ny,typeof(xscaler),typeof(yscaler)}(xscaler, yscaler, gps)
end
