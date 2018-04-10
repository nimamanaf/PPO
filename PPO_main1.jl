
include("envs.jl")
include("policy_fn.jl")
include("utils.jl")

using ArgParse, Knet, JLD

function main(args=ARGS)
    s = ArgParseSettings()
    s.description = "PPO main"
    @add_arg_table s begin
        ("--env_id"; arg_type=String; default="SpaceInvaders-v0"; help="gym environment id")
        ("--clip"; arg_type=Float64 ; default=0.1; help=" Initial Clipping parameter ")
        ("--cli_func";arg_type=String; default="linear"; help="clip range schedule function, {linear, constant}")
        ("--entcoeff"; arg_type=Float64 ; default=0.01 ; help="Entropy coefficint")
        ("--vfcoeff"; arg_type=Float64 ; default=1. ; help="Value function coefficint")
        ("--λ"; arg_type=Float64 ; default=0.95 ; help=" Generelized advantage estimation parameter")
        ("--adam_step_size"; arg_type=Float64 ; default=0.1 ; help=" Adam Step Size")
        ("--num_actors"; arg_type=Int ; default=1 ; help=" Number of parallel actors")
        ("--opt_epochs"; arg_type=Int; default=3; help="optimization epochs between environment interaction")
        ("--total_steps"; arg_type=Int; default=Int(10e6); help="total number of environment steps to take")
        ("--sequence-steps"; arg_type=Int; default=32; help="steps per sequence (for backprop through time)")
        ("--minibatch-steps"; arg_type=Int; default=256; help="steps per optimization minibatch")
        ("--lr-func"; arg_type=String; default="linear"; help="learning rate schedule function, {linear, constant}")
        ("--lr"; arg_type=Float64; default=2.5e-4; help="Initial learning rate")
        ("--γ"; arg_type=Float64; default=0.99; help="discount factor")
        ("--bs"; arg_type=Int; default=32; help="batch size")
        ("--atype";default=(gpu()>=0 ? "KnetArray{Float32}" : "Array{Float32}"))
        ("--worker_steps"; arg_type=Int; default=128; help="steps per worker between optimization rounds")

    end
    isa(args, AbstractString) && (args=split(args))
    o = parse_args(args, s)
    isa(args, AbstractString) && (args=split(args))
    if in("--help", args) || in("-h", args)
        ArgParse.show_help(s; exit_when_done=false)
        return
    end

    o["atype"] = eval(parse(o["atype"]))
    srand(1)
    env = GymEnv(o["env_id"], make_env)
    seed!(env, 1)
    inp = env.observation_space.shape
    num_actions = env.action_space.n
    h = (inp, 64, 32)
    W = π_init(h, num_actions)
    x = reset!(env)
    π, v =  policy_fn(W, x)
    
    seg = interact(W_old, env, o["worker_steps"], o)
    seg = add_vtarg_and_adv(seg::Transition, o["γ"], o["λ"])

end
seg = main(ARGS)
