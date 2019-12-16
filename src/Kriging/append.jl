function Base.append!(
    krg::Kriging{N,M},
    x::AbstractMatrix,
    y::AbstractMatrix,
) where {N,M}
    xs = krg.xscaler * x
    ys = krg.yscaler * y
    for i in 1:M
        append!(krg.gps[i], xs, ys[i, :])
    end
end
