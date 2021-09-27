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
    logNoise::Real = log(eps())/2
)
    nx, mx = size(x)
    ny, my = size(y)
    if mx != my
        throw(ArgumentError("`x` and `y` must have same number of observations"))
    end
    xs = xscaler * x
    ys = yscaler * y
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


getx(krg::Kriging, i::Int) = krg.xscaler \ view(krg.gps[1].x, :, i)
getx(krg::Kriging) = krg.xscaler \ krg.gps[1].x

function gety(krg::Kriging{N,M}, i::Int) where {N,M}
    y = similar(krg.gps[1].y, M)
    @inbounds for j in 1:M
        y[j] = krg.gps[j].y[i]
    end
    inverse!(krg.yscaler, y)
    y
end

function gety(krg::Kriging{N,M}) where {N,M}
    nsamples = length(krg.gps[1].y)
    y = similar(krg.gps[1].y, M, nsamples)
    @inbounds for j in 1:nsamples
        for i in 1:M
            y[i, j] = krg.gps[i].y[j]
        end
    end
    inverse!(krg.yscaler, y)
    y
end

Base.getindex(krg::Kriging, i::Integer) = gety(krg, i), getx(krg, i)


function get_samples(krg::Kriging{N,M}) where {N,M}
    getx(krg), gety(krg)
end
