
using PyCall

#const gym = PyNULL()

#function __init__()
#    copy!(gym, pyimport("gym"))
#end

@pyimport gym
@pyimport baselines.common.atari_wrappers as aw
@pyimport baselines.common.vec_env.subproc_vec_env as subvecenv
@pyimport baselines.common.vec_env.vec_frame_stack as vecfstak

env_id = "SpaceInvaders-v0"
id = "SpaceInvaders-v0"



const _gym_spaces = ["Box", "Discrete", "MultiDiscrete", "MultiBinary", "Tuple", "Dict"]

abstract type absSpace end
sample(space::absSpace) = error("sample function for $(typeof(space)) is missing")

mutable struct DiscreteS <: absSpace
    n
end
sample(s::DiscreteS) = rand(0:(s.n-1))

mutable struct BoxS <: absSpace
    low
    high
    shape
end

function sample(s::BoxS)
    if !isa(s.low, Array{Float32})#Float?
        r = rand() * (s.high - s.low) - s.low
        return r
    elseif length(s.low) == 1
        r = rand(s.shape...) * (s.high[1] - s.low[1]) + s.low[1]
        return r
    elseif size(s.low) == size(s.high)
        r = rand(s.shape...) .* (s.high - s.low) .+ s.low
        return r
    else
        error("Dimension mismatch")
    end
end

mutable struct TupleS <: absSpace
    spaces
end
sample(s::TupleS) = map(sample, s.spaces)

function julia_space(ps)
    class_name = ps[:__class__][:__name__]
    if class_name == "Discrete"
        return DiscreteS(ps[:n])
    elseif class_name == "Box"
        return BoxS(ps[:low], ps[:high], ps[:shape])
    elseif class_name == "Tuple"
        spaces = map(julia_space, ps[:spaces])
        return TupleS((spaces...))
    else
        error("$class_name has not been supported yet")
    end
end

struct Spec
    id
    trials
    reward_threshold
    nondeterministic
    tags
    max_episode_steps
    timestep_limit
end

mutable struct GymEnv
    name
    spec
    action_space
    observation_space
    reward_range
    gymenv
end

function make_atari(env_id)
    env = gym.make(env_id)
    env = aw.NoopResetEnv(env, noop_max=30)
    env = aw.MaxAndSkipEnv(env, skip=4)
    return env

end

function make_env(env_id, rank, seed)
    env = make_atari(env_id)
    env[:seed](seed+rank)
    env = aw.wrap_deepmind(env, episode_life=false, clip_rewards=false)
    return env
end

gymenv = make_env(id, 1, 0)
function GymEnv(id::String, make_env)
    try
        gymenv = make_env(id, 1, 0)
        #gymenv = gym[:make](id)
        spec = Spec(gymenv[:spec][:id],
                    gymenv[:spec][:trials],
                    gymenv[:spec][:reward_threshold],
                    gymenv[:spec][:nondeterministic],
                    gymenv[:spec][:tags],
                    gymenv[:spec][:max_episode_steps],
                    gymenv[:spec][:timestep_limit]
                   )
        action_space = julia_space(gymenv[:action_space])
        observation_space = julia_space(gymenv[:observation_space])

        env = GymEnv(id, spec, action_space,
                     observation_space, gymenv[:reward_range], gymenv)
        return env
    catch
        error("$id is missing")
    end
end

reset!(env::GymEnv) = env.gymenv[:reset]()
function render(env::GymEnv; mode="human")
    env.gymenv[:render](mode)
end

function step!(env::GymEnv, action)
    ob, reward, done, information = env.gymenv[:step](action)
    return ob, reward, done, information
end

close!(env::GymEnv) = env.gymenv[:close]()

seed!(env::GymEnv, seed=nothing) = env.gymenv[:seed](seed)



env_fns = Any[]
for rank in 1:num_actors
    push!(env_fns, make_env(env_id, rank, seed))
end
venv =  subvecenv.SubprocVecEnv(env_fns) # Not rernder able
venv = vecfstak.VecFrameStack(venv, 4)

test_env = make_env(env_id, 0, seed)
test_env = aw.FrameStack(test_env, 4)

jvenv = map(GymEnv(id, make_env), venv)

env = GymEnv(id, make_env)
