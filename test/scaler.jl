@testset "Scaler" begin

    S = randn(5, 100)
    scaler = StandardScaler(S)
    Ss = scaler * S
    @test isapprox(mean(Ss, dims = 2), zeros(5), atol = 10 * eps())
    @test isapprox(std(Ss, dims = 2), ones(5))
    Si = scaler \ Ss
    @test isapprox(S, Si)
    S = [-5.0 5.0]
    scaler = MinMaxScaler(S)
    Ss = scaler * S
    @test isapprox([0.0 1.0], Ss)
    @test isapprox(S, scaler \ Ss)
    scaler = MinMaxScaler([-10.0], [10.0])
    Ss = scaler * S
    @test isapprox([0.25 0.75], Ss)
    @test isapprox(S, scaler \ Ss)
end
