
mutable struct Transition
    ob
    rew
    vpred
    _new
    ac
    prevac
    nextvpred
    ep_rets
    ep_lens
    adv
    tdlamret
end

function traj_segment_generator(env, horizon)
    """
    Returns:
        obs (ArgumentDefaultsHelpFormatternsor): observations shaped [T + 1 x N x ...]
        rewards (FloatTensor): rewards shaped [T x N x 1]
        masks (FloatTensor): continuation masks shaped [T x N x 1]
            zero at done timesteps, one otherwise
        actions (LongTensor): discrete actions shaped [T x N x 1]
        steps (int): total number of steps taken
    """
    t = 1
    ac = sample(env.action_space) # not used, just so we have the datatype
    _new = true # marks if we're on first timestep of an episode
    ob = reset!(env)

    cur_ep_ret = 0 # return in current episode
    cur_ep_len = 0 # len of current episode
    ep_rets = Any[] # returns of completed episodes in this segment
    ep_lens = Any[] # lengths of ...

    # Initialize history arrays
    obs = Array([ob for _ in 0:horizon])
    rews = b0(horizon)
    vpreds = b0(horizon)
    news = b0(horizon)
    acs = Array([ac for _ in 1:horizon])
    prevacs = copy(acs)

    while true
        prevac = ac
        ac = sample(env.action_space) # take the actions with respcet to the policy here
        vpred = 0.1
        #ac, vpred = pi.act(stochastic, ob)
        # Slight weirdness here because we need value function at time T
        # before returning segment [0, T-1] so we get the correct
        # terminal value
        if t > 0 && t % horizon == 0
            return Transition(obs/255., rews, vpreds, news, acs, prevacs, vpred * (1 - _new),
             ep_rets, ep_lens, 0, 0)
            # Be careful!!! if you change the downstream algorithm to aggregate
            # several of these batches, then be sure to do a deepcopy
            ep_rets = []
            ep_lens = []
        end
        i = t
        obs[i] = ob
        vpreds[i] = vpred
        news[i] = _new
        acs[i] = ac
        prevacs[i] = prevac

        ob, rew, _new, _ = step!(env, ac)
        rews[i] = rew

        cur_ep_ret += rew
        cur_ep_len += 1
        if _new
            push!(ep_rets, cur_ep_ret)
            push!(ep_lens, cur_ep_len)
            cur_ep_ret = 0
            cur_ep_len = 0
            ob = reset!(env)
        end
        t += 1
    end
end


function add_vtarg_and_adv(seg::Transition, γ, λ)
    """
    Compute target value using TD(lambda) estimator, and advantage with GAE(lambda)
    """
    _new = append!(seg._new, 0) # last element is only used for last vtarg, but we already zeroed it if last new = 1
    vpred = append!(seg.vpred, seg.nextvpred)
    T = length(seg.rew)
    seg.adv = gaelam = b0(T)
    rew = seg.rew
    lastgaelam = 0
    for t in T:1            #reversed(range(T)):
        nonterminal = 1-_new[t+1]
        δ = rew[t] + γ * vpred[t+1] * nonterminal - vpred[t]
        gaelam[t] = lastgaelam = δ + γ * λ* nonterminal * lastgaelam
    end
seg.tdlamret = seg.adv + seg.vpred
end
