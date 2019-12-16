

function predict_full(krg::Kriging{N,M}, x::AbstractMatrix) where {N,M}
    if size(x, 1) != N
        throw(ArgumentError("`krg` and `x` do not have consistent dimensions"))
    end
    nobs = size(x, 2)
    y = similar(x, M, nobs)
    σ2 = similar(y)
    @inbounds for i in 1:M
        yr, sr = GP.predict_f(krg.gps[i], krg.xscaler * x)
        y[i, :] .= yr
        σ2[i, :] .= sr
    end
    inverse!(krg.yscaler, y)
    inverse_variance!(krg.yscaler, σ2)
    y, σ2
end

function predict_full(krg::Kriging, x::AbstractVector)
    y, σ2 = predict_full(krg, reshape(x, :, 1))
    vec(y), vec(σ2)
end
