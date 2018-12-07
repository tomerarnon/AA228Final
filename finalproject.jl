using Plots
using Distributions
using StatsBase, Statistics, LinearAlgebra, Random
using ParticleFilters, BeliefUpdaters
using POMDPs, POMDPModels, POMDPPolicies, POMDPModelTools, POMDPSimulators
using ARDESPOT, POMCPOW, BasicPOMCP, QMDP, MCTS, AEMS
using Printf

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
    ordered_states,
    actionindex



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
    r_fault::Float64
    r_repair::Float64
end

reliability(p::TruckMaintenance, s::TruckState) = exp(-s.d/ p.λ)
rand_distance(p::TruckMaintenance, rng::AbstractRNG = Random.GLOBAL_RNG) = 3000.0*rand(rng) + 5.0

Base.getindex(D::Dict, s::TruckState) = D[s.fault]

# random Normal constructor
function TruckMaintenance(n_sensors::Int
                          ;
                          λ = 100_000,
                          r_fault = 100,
                          r_repair = 50)

    sensor_dict = Dict{Bool, Vector{Normal}}()
    sensor_dict[true]   = Vector{Normal}()
    sensor_dict[false] = Vector{Normal}()
    for i in 1:n_sensors
        μ = 5*rand()
        σ = 10*rand()

        push!(sensor_dict[false], Normal(μ, σ))

        ### make slightly different distributions for :fault state
        # for now, just pick values of 5, 4 for no reason
        μ += 5*(rand()-1)  # ± 2.5
        σ *= 4*rand()        # scale by 0-4
        push!(sensor_dict[true], Normal(μ, σ))
    end
    TruckMaintenance(λ, sensor_dict,r_fault, r_repair)
end


initial_state(p::TruckMaintenance) = TruckState(false, 0)
discount(::TruckMaintenance) = 0.95

n_actions(p::TruckMaintenance) = 2
n_states(p::TruckMaintenance)  = 2

ordered_states(p::TruckMaintenance)                        = states(p)
initialstate_distribution(p::TruckMaintenance)             = SparseCat(states(p), [1.0, 0.0])

states(p::TruckMaintenance)                                = [TruckState(false, 0.0),     TruckState(true, 0.0)]
states(p::TruckMaintenance, s::TruckState)                 = [TruckState(false, s.d),     TruckState(true, s.d)]
states(p::TruckMaintenance, s::TruckState, a::TruckAction) = [TruckState(false, s.d+a.d), TruckState(true, s.d+a.d)]

actions(p::TruckMaintenance)                               = [TruckAction(false, rand(50:50:1500)), TruckAction(true, 0)]

stateindex(p::TruckMaintenance, s::TruckState)   = 2 - s.fault
actionindex(p::TruckMaintenance, a::TruckAction) = 2 - a.repair

# function POMDPSimulators.get_initialstate(sim::Simulator, initialstate_dist::BoolDistribution)
    # f = rand(sim.rng, initialstate_dist)
    # return TruckState(f, 0.0)
# end


function generate_sr(p::TruckMaintenance, s::TruckState, a::TruckAction, rng::AbstractRNG = Random.GLOBAL_RNG)
    sp = generate_s(p, s, a, rng)
    r = reward(p, s, a, sp)

    return sp, r
end

function reward(p::TruckMaintenance, s::TruckState, a::TruckAction, sp::TruckState)
    r = 0.0

    r -= sp.fault ? p.r_fault : 0
    r -= a.repair ? p.r_repair : 0

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
# function generate_o(p::TruckMaintenance, s::TruckState, a::TruckAction, sp::TruckState, rng::AbstractRNG = Random.GLOBAL_RNG) # only sp matters
#     fault_dists   = p.sensor_dict[true]
#     nofault_dists = p.sensor_dict[false]
#
#     o_vec = zeros(Float64, length(fault_dists))
#
#     o_vec = [rand(rng, x) for x in p.sensor_dict[sp]]
#     if sp.fault
#         for i in 1:length(o_vec)
#             o_vec[i] = rand(p.sensor_dict[sp][i])
#         end
#         return o_vec
#     end
#
#     for i in 1:length(o_vec)
#         x = randn(rng)
#
#         fault_obs   = scl(fault_dists[i], x)
#         nofault_obs = scl(nofault_dists[i], x)
#
#         o_vec[i] = lerp(fault_obs, nofault_obs, 0.9*reliability(p, sp)) # when reliability is close to 1, the value is close to nofault but never reaches it
#     end
#     o_vec
# end

function generate_o(p::TruckMaintenance, s::TruckState, a::TruckAction, sp::TruckState, rng::AbstractRNG = Random.GLOBAL_RNG) # only sp matters
    return [rand(rng, d) for d in observation(p,sp)]
end

function observation(p::TruckMaintenance, sp::TruckState)

    new_distributions = similar(p.sensor_dict[sp])

    for i in 1:length(new_distributions)

        df  = p.sensor_dict[true][i]
        dnf = p.sensor_dict[false][i]

        # interpolate between fault and nofault according to reliability:
        rel = reliability(p, sp)
        lerped_μ = lerp(dnf.μ, df.μ, 0.9*rel)
        lerped_σ = lerp(dnf.σ, df.σ, 0.9*rel)

        new_distributions[i] = Normal(lerped_μ, lerped_σ)
    end

    return new_distributions
end

# returns probabilty of entering :fault state
function transition(p::TruckMaintenance, s::TruckState, a::TruckAction)
    # states(p, ...) should return false/true always (worth double checking always)
    a.repair && return SparseCat(states(p), [1.0, 0.0])
    # elseif:
    s.fault && return SparseCat(states(p, s, a), [0.0, 1.0])
    # else:
    rel = reliability(p, s)
    return SparseCat(states(p, s, a), [rel, 1-rel])
end

scl(d::Normal, x) = @. x*d.σ + d.μ
lerp(a, b, x) = a + x*(b-a)

# TYPE PIRACY
Distributions.pdf(Nvec::Vector{<:Normal}, x) = prod(pdf.(Nvec, x))
POMDPs.pdf(d::BoolDistribution, s::TruckState) = pdf(d, s.fault)

# p = TruckMaintenance(10)
# belief_updater = SIRParticleFilter(p, 100)
# lst_solver = [BeliefMCTSSolver(DPWSolver(), belief_updater), POMCPSolver(), DESPOTSolver(), POMCPOWSolver(), QMDPSolver()]
# LstTime = []
# util = []


# takes in a vector of literal solvers (i.e. like lst_solver above)
# returns a dictionary of dictionaries
# the outer dictionary are accessed with with solver literal
# the inner dictionaries have keys:
# :T         => average time
# :U         => average reward
# :rU        => average reward of a random policy
# :histories => all of the recorded histories as a vector
function simulate_a_bunch(solvers
                            ;
                          n_particles = 100,
                          n_sensors = 10,
                          n_sims = 50,
                          max_steps = 100,
                          λ = 100_000)

    T_U_R_H_dict() = Dict(:T=>0.0, :U=>0.0, :rU=>0.0, :histories=>Vector{POMDPHistory}())
    T_U_R_hists = Dict(sol=>T_U_R_H_dict() for sol in solvers)

    for sol in solvers
        println(sol)
        t = total_reward = total_random_reward =  0.0
        for i in 1:n_sims
            print(i, ",")

            # just in case some shenanigans go on that might mess things up,
            # deepcopy the solver every time you use it.
            use_sol = deepcopy(sol)
            # recreate p each time so that the sensors values are randomized
            p = TruckMaintenance(n_sensors)
            belief_updater = SIRParticleFilter(p, n_particles)
            policy = solve(use_sol, p)

            # time the main runtime
            t_start = time_ns()
            history = simulate(HistoryRecorder(max_steps = 100), p, policy, belief_updater)
            t_end   = time_ns()
            # also run a random version
            random_history = simulate(HistoryRecorder(max_steps = 100), p, RandomPolicy(p), belief_updater)

            total_reward += sum(history.reward_hist)
            total_random_reward += sum(random_history.reward_hist)

            t += Float64(t_end - t_start) / 1E9
            push!(T_U_R_hists[sol][:histories], history)
        end
        T_U_R_hists[sol][:T] = t/n_sims
        T_U_R_hists[sol][:U] = total_reward/n_sims
        T_U_R_hists[sol][:rU] = total_random_reward/n_sims
    end

    return T_U_R_hists
end

# policy = solve(lst_solver[5], p)
# history = simulate(HistoryRecorder(max_steps = 100), p, policy, belief_updater)
# total = 0
# k = 0
# for (s, a, r, sp) in eachstep(history, "(s, a, r, sp)")
#     println("s => $(s.fault), $(s.d), \t a => $(a.repair), $(a.d)")
#     global total = total + r
#     push!(util, total)
#     if s.fault
#         global k = k + 1
#     end
# end
# plot(util)
# @show k

# policy = solve(lst_solver[1], p)
# history = simulate(HistoryRecorder(max_steps = 100), p, policy, belief_updater)
# println("Total discounted reward: $(discounted_reward(history))")
# policy = RandomPolicy(p)
# history = simulate(HistoryRecorder(max_steps = 100), p, policy, belief_updater)
# println("Total discounted reward: $(discounted_reward(history))")

# for solver in lst_solver
#     policy = solve(solver, p)
#     avg = 0
#     N = 10
#     t1 = time_ns()
#     for i = 1:N
#         @show i
#         history = simulate(HistoryRecorder(max_steps = 100), p, policy, belief_updater)
#         avg = avg + discounted_reward(history)
#     end
#     avg = avg/N
#     t2 = time_ns()
#     elap = t2 - t1
#     push!(util,avg)
#     push!(LstTime,elap)
# end


# Lst = []
# for i = 10:90
#     p = TruckMaintenance(10,r_repair = i)
#     belief_updater = SIRParticleFilter(p, 100)
# # belief_updater = DiscreteUpdater(p)
#     # solver = POMCPSolver()
#     solver = BeliefMCTSSolver(DPWSolver(), belief_updater)
#     policy = solve(solver, p)
#     try
#         history = simulate(HistoryRecorder(max_steps = 100), p, policy, belief_updater)
#         println("Total discounted reward: $(discounted_reward(history))")
#         lst = []
#         for (s, a, sp) in eachstep(history, "s,a,sp")
#             if a.repair
#                 push!(lst,s.d)
#             end
#         end
#         push!(Lst, mean(lst))
#     catch err
#     end
# end

# s = state_hist(history)[1]
# a = action(policy, s)

# p = TruckMaintenance(10,λ = 100000)
# solver = POMCPSolver(tree_queries=5, c=10.0, rng=MersenneTwister(1))
# planner = solve(solver, p)
# a, info = action_info(planner, initialstate_distribution(p), tree_in_info=true)

# inchrome(D3Tree(info[:tree], init_expand=5))

# p = TruckMaintenance(10,λ = 100000)
# belief_updater = SIRParticleFilter(p, 100)
# # belief_updater = DiscreteUpdater(p)
# # solver = POMCPSolver()
# solver = BeliefMCTSSolver(DPWSolver(), belief_updater)
# policy = solve(solver, p)
# for i = 1:50
#     p = TruckMaintenance(10,λ = 100000)
#     belief_updater = SIRParticleFilter(p, 100)
#     solver = BeliefMCTSSolver(DPWSolver(), belief_updater)
#     policy = solve(solver, p)
#     history = simulate(HistoryRecorder(max_steps = 10), p, policy, belief_updater)
#     tree = policy.tree
#     for (s, b, a, o) in eachstep(history, "sbao")
#         println("State was $s,")
#         println("action $a was taken,")
#         println("and observation $o was received.\n")
#     end
#     println("MCTS Policy: Discounted reward was $(discounted_reward(history)).")

#     policy = RandomPolicy(p)
#     history = simulate(HistoryRecorder(max_steps = 10), p, policy, belief_updater)

#     tree = policy.tree
#     for (s, b, a, o) in eachstep(history, "sbao")
#         println("State was $s,")
#         println("action $a was taken,")
#         println("and observation $o was received.\n")
#     end
#     println("Random Policy: Discounted reward was $(discounted_reward(history)).")
#     push!(reward2,discounted_reward(history))
# end
# plot(reward1)
# n_sensors = 3
# for i in 1:n_sensors
#     μ = 10*rand()
#     σ = 10*rand()
#     display(display(plot(Normal(μ,σ),lw=3)))

#     ### make slightly different distributions for :fault state
#     # for now, just pick values of 5, 2 for no reason
#     μ += 5*(rand()-1)  # ± 2.5
#     σ *= 2*rand()        # scale by 0-2
#     display(display(plot!(Normal(μ ,σ),lw=3)))
# end
