using POMDPs
using StaticArrays
using Distributions
using StatsBase
using Statistics
using LinearAlgebra
using Random
using ParticleFilters
using POMDPSimulators
using POMDPModelTools
using BeliefUpdaters

#import just about everything
import POMDPs: initial_state,
    discount,
    n_states,
    n_actions,
    states,
    actions,
    initialstate_distribution,
    reward,
    generate_s,
    generate_o,
    generate_sr,
    observation,
    transition,
    stateindex,
    ordered_states



"""
TruckState
    fault - true/false
    d     - distance since last repair
"""
struct TruckState
    fault::Bool
    d::Float64
end

"""
TruckAction
    repair - true/faluse
    d      - distance of this drive segment
"""
struct TruckAction
    repair::Bool
    d::Float64
end

# """
# TruckObs
# """
# struct TruckObs
#     o_vec::Vector{Float64}
# end

"""
TruckMaintenance

    λ::Float  # mean failure rate
    sensor_dict::Dict{Symbol, Vector{<:Distribution}}


Below is how sensor_dict is arranged. Each entry in sensor_dict[:fault] is a
distribution of a sensor measurement in the :fault state. sensor_dict[:nofault]
is the corresponding distribution for the nofault case

    sensor_dict:                     # Dict
        :fault -                       # Vector
            [1]: Normal(μ, σ)          # any Distribution type
            [2]: Normal(μ, σ)
            ...

        :nofault -                     # Vector
            [1]: Normal(μ, σ)          # any Distribution type
            [2]: Normal(μ, σ)
            ...
"""
struct TruckMaintenance <: POMDP{TruckState, TruckAction, Vector{Float64}}
    λ::Float64
    sensor_dict::Dict{Bool, Vector{<:Distribution}} # consider parameterizing the type
end

reliability(p::TruckMaintenance, s::TruckState) = exp(-s.d/ p.λ)
rand_distance(p::TruckMaintenance, rng::AbstractRNG  = Random.GLOBAL_RNG= GLOBAL_RNG) = 3000.0*rand(rng) + 5.0

Base.getindex(D::Dict, s::TruckState) = D[s.fault]

# random Normal constructor
function TruckMaintenance(n_sensors::Int; λ = 10_000)

    sensor_dict = Dict{Bool, Vector{Normal}}()
    sensor_dict[true]   = Vector{Normal}()
    sensor_dict[false] = Vector{Normal}()
    for i in 1:n_sensors
        μ = 10*rand()
        σ = 10*rand()

        push!(sensor_dict[false], Normal(μ, σ))

        ### make slightly different distributions for :fault state
        # for now, just pick values of 5, 2 for no reason
        μ += 5*(rand()-0.5)  # ± 2.5
        σ *= 2*rand()        # scale by 0-2
        push!(sensor_dict[true], Normal(μ, σ))
    end
    TruckMaintenance(λ, sensor_dict)
end


initial_state(p::TruckMaintenance) = TruckState(false, 0)
discount(::TruckMaintenance) = 0.9

n_states(p::TruckMaintenance)  = 2
n_actions(p::TruckMaintenance) = 2
ordered_states(p::TruckMaintenance)        = [TruckState(false, 0.0), TruckState(true, 0.0)]
states(p::TruckMaintenance, s::TruckState) = [TruckState(false, s.d), TruckState(true, s.d)]
actions(p::TruckMaintenance)               = [TruckAction(false, 1000), TruckAction(true, 1000)]
stateindex(p::TruckMaintenance, s::TruckState) = 2 - s.fault
initialstate_distribution(p::TruckMaintenance) = BoolDistribution(0.0)

function POMDPSimulators.get_initialstate(sim::Simulator, initialstate_dist::BoolDistribution)
    f = rand(sim.rng, initialstate_dist)
    return TruckState(f, 0.0)
end


function reward(p::TruckMaintenance, s::TruckState, a::TruckAction, sp::TruckState)
    r = 0.0

    r -= sp.fault ? 100 : 0
    r -= a.repair ?  10 : 0

    return r
end

function generate_s(p::TruckMaintenance, s::TruckState, a::TruckAction, rng::AbstractRNG = Random.GLOBAL_RNG)
    if a.repair
        return TruckState(false, 0)
    end

    if s.fault
        return TruckState(true, s.d)
    end
    # ∴ s is :nofault —
    r = rand()
    if r > reliability(p, s)
        return TruckState(true, s.d+a.d)
    else
        return TruckState(false, s.d+a.d)
    end
end

# assumes each sensor measurement is independent and does not depend on previous measurements
# (maybe they should be linked to previous state for smooth transitions?)
# function generate_o(p::TruckMaintenance, s::TruckState, a::Bool, sp::TruckState) # only sp matters
#     o_vec = zeros(Float64, length(p.sensor_dict[sp]))
#     for i in 1:length(o_vec)
#         o_vec[i] = rand(p.sensor_dict[sp][i])
#     end
#     o_vec
# end


# assumes each sensor measurement is independent and does not depend on previous measurements
function generate_o(p::TruckMaintenance, s::TruckState, a::TruckAction, sp::TruckState, rng::AbstractRNG = Random.GLOBAL_RNG) # only sp matters
    fault_dists   = p.sensor_dict[true]
    nofault_dists = p.sensor_dict[false]

    o_vec = zeros(Float64, length(fault_dists))

    o_vec = [rand(rng, x) for x in p.sensor_dict[sp]]
    # if sp.fault
        # for i in 1:length(o_vec)
        #     o_vec[i] = rand(p.sensor_dict[sp][i])
        # end
        # return o_vec
    # end

    # for i in 1:length(o_vec)
    #     x = randn(rng)

    #     fault_obs   = scl(fault_dists[i], x)
    #     nofault_obs = scl(nofault_dists[i], x)

    #     o_vec[i] = lerp(fault_obs, nofault_obs, 0.9*reliability(p, sp)) # when reliability is close to 1, the value is close to nofault but never reaches it
    # end
    # o_vec
end

function generate_sr(p::TruckMaintenance, s::TruckState, a::TruckAction, rng::AbstractRNG = Random.GLOBAL_RNG)
    sp = generate_s(p, s, a, rng)
    r = reward(p, sp, a)

    return sp, r
end


# The other formulation method:

function observation(p::TruckMaintenance, sp::TruckState)
    return p.sensor_dict[sp]
end

# returns probabilty of entering :fault state
function transition(p::TruckMaintenance, s::TruckState, a::TruckAction)
    a.repair && return BoolDistribution(0.0)
    # elseif:
    s.fault && return BoolDistribution(1.0)
    # else:
    rel = reliability(p, s)
    return BoolDistribution(1-rel)
end

scl(d::Normal, x) = x.*d.σ .+ d.μ
lerp(a, b, x) = a + x*(b-a)

# TYPE PIRACY
Distributions.pdf(Nvec::Vector{<:Normal}, x) = prod(pdf.(Nvec, x))
POMDPs.pdf(d::BoolDistribution, s::TruckState) = pdf(d, s.fault)



# using MCTS, BasicPOMCP, QMDP

# p = TruckMaintenance(3)
# belief_updater = SIRParticleFilter(p, 100)
# # belief_updater = DiscreteUpdater(p)

# solver = POMCPSolver()
# solver = BeliefMCTSSolver(DPWSolver(), belief_updater)
# policy = solve(solver, p)

# history = simulate(HistoryRecorder(max_steps = 20), p, policy, belief_updater)