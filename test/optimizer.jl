@testset "Optimizer" begin
    @testset "Global" begin
        f(x) =
            (x[2] - 5.1 / (4 * π^2) * x[1]^2 + 5 / π * x[1] - 6)^2 +
            10 * (1 - 1 / (8π)) * cos(x[1]) +
            10 +
            5 * x[1]
        optix = [-3.689285272296118, 13.629987729088747]
        optif = -16.64402157084319
        lb = [-5.0, 0.0]
        ub = [10.0, 15.0]
        for o in (CRSOptimizer, ISRESOptimizer)
            opt = o(maxeval = 10000, population = 100)
            fx, x, ret = minimize(opt, f, :Auto, lb, ub, (lb + ub) / 2)
            @test isapprox(fx, optif, rtol = 1e-3)
            @test isapprox(x, optix, rtol = 1e-3)
        end

    end
    @testset "Local" begin
        f(x) = sum(i -> i * x[i]^2, length(x))
        lb = fill(-10.0, 5)
        ub = fill(10.0, 5)
        optif = 0.0
        optix = zeros(5)
        for o in (LBFGSOptimizer, BOBYQAOptimizer)
            opt = o(maxeval = 1000)
            fx, x, ret = minimize(opt, f, :Auto, lb, ub, (lb + ub) / 2)
            @test isapprox(fx, optif, rtol = 1e-3)
            @test isapprox(x, optix, rtol = 1e-3)
        end
    end
end
