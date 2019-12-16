
function filt_zeros!(zoomer::AbstractArray{T}) where {T<:AbstractFloat}
    for i in eachindex(zoomer)
        if isapprox(zoomer[i], zero(T), atol = eps(T))
            zoomer[i] = one(T)
        end
    end
end

struct LinearScaler{N,V<:AbstractVector} <: Scaler{N}
    offset::V
    zoomer::V
end

function StandardScaler(
    mean::AbstractVector{T},
    std::AbstractVector{T},
) where {T<:AbstractFloat}
    if length(mean) != length(std)
        throw(DimensionMismatch("`mean` and `std` must have the same length"))
    end
    μ = copy(mean)
    σ = copy(std)
    filt_zeros!(σ)
    LinearScaler{length(μ),typeof(μ)}(μ, σ)
end

function StandardScaler(x::AbstractMatrix{T}) where {T<:AbstractFloat}
    n, m = size(x)
    if m < 2
        throw(ArgumentError("`x` must have at least 2 samples (cols)"))
    end
    μ = similar(x, n)
    mean!(μ, x)
    σ = reshape(std(x, dims = 2, mean = μ), :)
    filt_zeros!(σ)
    LinearScaler{length(μ),typeof(μ)}(μ, σ)
end

function MinMaxScaler(
    min::AbstractVector{T},
    max::AbstractVector{T},
) where {T<:AbstractFloat}
    if length(min) != length(max)
        throw(DimensionMismatch("`min` and `max` must have the same length"))
    end
    dlt = max - min
    filt_zeros!(dlt)
    LinearScaler{length(dlt),typeof(dlt)}(copy(min), dlt)
end

function MinMaxScaler(x::AbstractMatrix{T}) where {T<:AbstractFloat}
    n, m = size(x)
    if m < 2
        throw(ArgumentError("`x` must have at least 2 samples (cols)"))
    end
    min = similar(x, n)
    minimum!(min, x)
    max = similar(min)
    maximum!(max, x)
    @. max -= min
    filt_zeros!(max)
    LinearScaler{length(max),typeof(max)}(min, max)
end

transform!(s::LinearScaler, x::AbstractVecOrMat) =
    @. x = (x - s.offset) / s.zoomer

transform(s::LinearScaler, x::AbstractVecOrMat) = transform!(s, copy(x))

inverse!(s::LinearScaler, x::AbstractVecOrMat) =
    @. x = (x * s.zoomer) + s.offset

inverse(s::LinearScaler, x::AbstractVecOrMat) = inverse!(s, copy(x))

Base.:*(s::LinearScaler, x::AbstractVecOrMat) = transform(s, x)

Base.:\(s::LinearScaler, x::AbstractVecOrMat) = inverse(s, x)

inverse_variance!(s::LinearScaler, v::AbstractVecOrMat) = @. v *= s.zoomer^2
transform_variance!(s::LinearScaler, v::AbstractVecOrMat) = @. v /= s.zoomer^2

Base.eltype(s::LinearScaler) = eltype(s.offset)