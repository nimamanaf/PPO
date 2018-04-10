N = 7

addprocs(N)
num_actors = N
using ArgParse, Knet, AutoGrad

include("envs.jl")
include("policy_fn.jl")
include("utils.jl")

function main(args=ARGS)
    s = ArgParseSettings()
    s.description=" Actor-Critic algorithm."
    @add_arg_table s begin
        ("--env_id"; arg_type=String; default="SpaceInvaders-v0"; help="gym environment id")
        ("--clip"; arg_type=Float32 ; default=Float32(0.1); help=" Initial Clipping parameter ")
        ("--cli_func";arg_type=String; default="linear"; help="clip range schedule function, {linear, constant}")
        ("--entcoeff"; arg_type=Float32 ; default=Float32(0.01) ; help="Entropy coefficint")
        ("--vfcoeff"; arg_type=Float32 ; default=Float32(1.) ; help="Value function coefficint")
        ("--λ"; arg_type=Float32 ; default=Float32(0.95) ; help=" Generelized advantage estimation parameter")
        ("--adam_step_size"; arg_type=Float32 ; default=Float32(0.1) ; help=" Adam Step Size")
        ("--num_actors"; arg_type=Int ; default=2 ; help=" Number of parallel actors")
        ("--opt_epochs"; arg_type=Int; default=3; help="optimization epochs between environment interaction")
        ("--total_steps"; arg_type=Int; default=Int(10e3); help="total number of environment steps to take")
        ("--sequence-steps"; arg_type=Int; default=32; help="steps per sequence (for backprop through time)")
        ("--minibatch-steps"; arg_type=Int; default=256; help="steps per optimization minibatch")
        ("--lr-func"; arg_type=String; default="linear"; help="learning rate schedule function, {linear, constant}")
        ("--lr"; arg_type=Float32; default=Float32(2.5e-4); help="Initial learning rate")
        ("--γ"; arg_type=Float32; default=Float32(0.99); help="discount factor")
        ("--bs"; arg_type=Int; default=32; help="batch size")
        ("--atype";default=(gpu()>=0 ? "KnetArray{Float32}" : "Array{Float32}"))
        ("--worker_steps"; arg_type=Int; default=128; help="steps per worker between optimization rounds")

    end

    srand(12345)
    isa(args, AbstractString) && (args=split(args))
    if in("--help", args) || in("-h", args)
        ArgParse.show_help(s; exit_when_done=false)
        return
    end

    o = parse_args(args, s)
    o["atype"] = eval(parse(o["atype"]))

    env = GymEnv(o["env_id"], make_env)
    venv = make_venv(o["num_actors"], o["env_id"])
    #seed!(env, 12345)

    inp = env.observation_space.shape
    num_actions = env.action_space.n

    #A3C paper
    h = ((8,8,16), (4,4,32), 256) 
    #w = init_weights(inp, 100, 100,  num_actions, o["atype"])
    w =  init_weights_shared(inp, h, num_actions, atype)

    opts = Dict()
    for k in keys(w)
        opts[k] = Adam(lr=o["lr"])
    end
    total_steps = o["total_steps"]
    w = play!(w, opts, env, venv, num_actions, total_steps, o)
        #println("episode $i , total rewards: $total")
end

main()

#end
