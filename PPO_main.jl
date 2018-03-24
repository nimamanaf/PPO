





function main(args=ARGS)
    s = ArgParseSettings()
    s.description = "PPO main"
    @add_arg_table s begin
        ("--timesteps_per_actorbatch"; arg_type= ; default= ; help="Timesteps per actor per update ") # the deafult ?
        ("--clip_param"; arg_type=Float32 ; default=0.1; help="Clipping parameter ")
        ("--entcoeff"; arg_type=Float32 ; default=0.01 ; help="Entropy coefficint")
        ("--vfcoeff"; arg_type=Float32 ; default=1. ; help="Value function coefficint")
        ("--λ"; arg_type=Float32 ; default=0.95 ; help=" Generelized advantage estimation parameter")
        ("--adam_step_size"; arg_type= ; default= ; help=" Adam Step Size")
        ("--num_actors"; arg_type=Int32 ; default=8 ; help="  ")
        ("--adam_step_size"; arg_type= ; default= ; help="  ")
        ("--adam_step_size"; arg_type= ; default= ; help="  ")

        ("--horizon"; arg_type=Int32; default=128; help="Espisode length (T)")
        ("--lr"; arg_type=Float64; default=0.001; help="learning rate")
        ("--gamma"; arg_type=Float64; default=0.99; help="discount factor")
        ("--hiddens"; arg_type=Int; nargs='+'; default=[32]; help="number of units in the hiddens for the mlp")
        ("--env_id"; default="CartPole-v0")
        ("--render"; action=:store_true)
        ("--memory"; arg_type=Int; default=1000; help="memory size")
        ("--bs"; arg_type=Int; default=32; help="batch size")
        ("--stack"; arg_type=Int; default=4; help="length of the frame history")
        ("--save"; default=""; help="model name")
        ("--load"; default=""; help="model name")
        ("--atype";default=(gpu()>=0 ? "KnetArray{Float32}" : "Array{Float32}"))
        ("--play"; action=:store_true; help="only play")
        ("--printinfo"; action=:store_true; help="print the training messages")
    end
    isa(args, AbstractString) && (args=split(args))
    o = parse_args(args, s)
    isa(args, AbstractString) && (args=split(args))
    if in("--help", args) || in("-h", args)
        ArgParse.show_help(s; exit_when_done=false)
        return
    end

    o["atype"] = eval(parse(o["atype"]))
    srand(12345)
    env = GymEnv(o["env_id"])
    seed!(env, 12345)

    INPUT = env.observation_space.shape[1] * o["stack"]
    OUTPUT = env.action_space.n

    if o["load"] == ""
        w = init_weights(INPUT, o["hiddens"], OUTPUT, o["atype"])
    else
        w = load_model(o["load"], o["atype"])
    end

    opts = Dict()
    for k in keys(w)
        opts[k] = Rmsprop(lr=o["lr"])
    end

    buffer = ReplayBuffer(o["memory"])

    exploration = PiecewiseSchedule([(0, 1.0),
                                     (round(Int, o["frames"]/5), 0.1),
                                     (round(Int, o["frames"]/3.5), 0.1)])

    rewards, frames = dqn_learn(w, opts, env, buffer, exploration, o)
end

PROGRAM_FILE == "dqn.jl" && main(ARGS)

end
