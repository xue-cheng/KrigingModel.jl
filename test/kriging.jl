@testset "Kriging" begin
    d, n = 3, 10
    x = 2π * rand(d, n)
    y1 = [sum(sin, view(x, :, i)) / d for i in 1:n]
    y2 = [sum(cos, view(x, :, i)) / d for i in 1:n]
    y = vcat(transpose(y1), transpose(y2))
    mZero = MeanZero()
    kern = SE(0.0, 0.0)
    ntest = 5
    xtest = randn(d, ntest)
    krg = Kriging(x, y, mZero, kern)
    @testset "predict samples" begin
        y_pred, sig2 = predict_full(krg, x)
        @test maximum(abs, y - y_pred) ≈ 0.0 atol = 0.1
    end
    @testset "predict tests" begin
        y_pred, sig2 = predict_full(krg, xtest)
    end
    @show krg.gps[1].target
    @show krg.gps[2].target
    tune!(krg, ([-10.0, -10.0], [10.0, 10.0]), nothing; global_eval = 10000)
end
