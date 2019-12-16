module KrigingModel

# using M:f  = call `f` only
# import M   = call `M.anyfunction`
# import M:f = extend `f`

using Statistics: mean!, mean, std
using ElasticArrays: ElasticArray

import GaussianProcesses, LinearAlgebra

const GP = GaussianProcesses

include("scaler/scaler.jl")
include("optimizer/optimizer.jl")
include("Kriging/Kriging.jl")

export StandardScaler, MinMaxScaler
export CRSOptimizer, ISRESOptimizer, LBFGSOptimizer, BOBYQAOptimizer
export minimize, maximize
export Kriging, tune!, predict_full, getx, gety

end # module
