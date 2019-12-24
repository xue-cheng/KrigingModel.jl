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
    yt1 = [sum(sin, view(xtest, :, i)) / d for i in 1:ntest]
    yt2 = [sum(cos, view(xtest, :, i)) / d for i in 1:ntest]
    ytest = vcat(transpose(yt1), transpose(yt2))
    krg = Kriging(x, y, mZero, kern)
    @testset "predict samples" begin
        y_pred, sig2 = predict_full(krg, x)
        @test maximum(abs, y - y_pred) ≈ 0.0 atol = 0.1
    end
    @testset "predict tests" begin
        y_pred, sig2 = predict_full(krg, xtest)
    end
    @testset "tune local" begin
        tune!(krg, ([-10.0, -10.0], [10.0, 10.0]), nothing; verbose = true)
    end
    @testset "tune" begin
        tune!(
            krg,
            ([-10.0, -10.0], [10.0, 10.0]),
            nothing;
            global_eval = 10000,
            verbose = true,
        )
    end
    @testset "append" begin
        append!(krg, xtest, ytest)
    end
    @testset "get" begin
        x1 = getx(krg, 1)
        y1 = gety(krg, 1)
        @test x1 ≈ x[:, 1] atol = 1e-6
        @test y1 ≈ y[:, 1] atol = 1e-6
    end
    @testset "get all" begin
        x1 = getx(krg)
        y1 = gety(krg)
        @test x1 ≈ hcat(x,xtest) atol = 1e-6
        @test y1 ≈ hcat(y,ytest) atol = 1e-6
    end
end
