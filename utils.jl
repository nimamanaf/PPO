
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


function π_softmax(π)
    probs = exp.(π) ./ sum(exp.(π), 1)
end

function sample_action(π, N) # take action with a softmax funcction
    probs = π_softmax(π)
    c_probs = cumsum(probs)
    x = c_probs .> rand(1, N)
    return map(n -> indmax(x[:,n])-1, 1:N)
end
@zerograd sample_action(π, N)

function interact(w, venv, num_actions, o)
    """
    Returns: Transition.
        obs (ArgumentDefaultsHelpFormatternsor): observations shaped [T + 1 x N x ...]
        rewards (FloatTensor): rewards shaped [T x N x 1]
        _new (FloatTensor): continuation masks shaped [T x N x 1]
            zero at done timesteps, one otherwise
        actions (LongTensor): discrete actions shaped [T x N x 1]
        ep_lens (int): total number of steps taken
    """
    N = o["num_actors"]
    T = o["worker_steps"]
    cur_ep_ret = 0 # return in current episode
    cur_ep_len = 0 # len of current episode
    ep_rets = Any[] # returns of completed episodes in this segment
    ep_lens = Any[] # lengths of ...

    # Initialize history arrays
    obs = zeros(Float32, 84, 84, 4, N, T+1)
    rews = zeros(Float32, N, T)
    vpreds = zeros(Float32, N, T)
    news = zeros(Float32, N, T)
    acs = zeros(Int, N, T)
    prevacs = copy(acs)
    _new = true # marks if we're on first timestep of an episode
    ob = pmap(x -> reset!(x), venv)
    #ob = pmap(x -> reshape(x, (84, 84, 4, 1)), ob)
    pmap(n -> obs[:,:,:,n, 1] = ob[n], 1:N)
    ac = rand(0:(num_actions-1), N)
    #π, vpred = predict(w, obs[:,:,:,:,1])
    #ac = sample_action(π, N)
    #vpreds[:, 1] = vpred
    #acs[:, 1] = ac
    #prevacs[:, 1] = prevac
    #for j = 1:N
    #    news[j, 1] = _new
    #end
    for i = 1:T
        prevacs[:, i] = ac

        π, vpred = predict(w, obs[:,:,:,:,i])
        ac = sample_action(π, N)
        vpreds[:, i] = vpred
        acs[:, i] = ac

        # Slight weirdness here because we need value function at time T
        # before returning segment [0, T-1] so we get the correct
        # terminal value
        x = pmap(n -> step!(venv[n], ac[n]), 1:N) #x[i]    ob, rew, _new, _
        #ob = pmap(n -> reshape(x[n][1], (84, 84, 4, 1)), 1:N)

        pmap(n -> obs[:,:,:,n, i+1] = x[n][1], 1:N)
        pmap(n -> rews[:, i] = x[n][2], 1:N)
        pmap(n -> news[:, i] = (1-x[n][3]), 1:N)

        #π, vpred = predict(w, obs[:,:,:,:,i+1])
        #ac = sample_action(π, N)
        #cur_ep_ret += rew
        cur_ep_len += 1
        #i += 1
        for e=1:N
            if news[e, i] != 1
                push!(ep_rets, cur_ep_ret)
                push!(ep_lens, cur_ep_len)
                cur_ep_ret = 0
                cur_ep_len = 0
                obs[:,:,:, e, i+1] = reset!(venv[e])
            end
        end
    end

    return Transition(obs/Float32(255.), rews, vpreds, news, acs, prevacs, vpreds[:, end] .* (1 - news[:, end]),
     ep_rets, ep_lens, [], [])
        # Be careful!!! if you change the downstream algorithm to aggregate
        # several of these batches, then be sure to do a deepcopy
        #ep_rets = []
        #ep_lens = []
end


function add_vtarg_and_adv(seg::Transition, o)
    """
    Compute target value using TD(lambda) estimator, and advantage with GAE(lambda)
    """
    γ =  o["γ"]
    λ =  o["λ"]
    N = o["num_actors"]
    T = o["worker_steps"]

    _news = hcat(seg._new, zeros(Float32, N)) # last element is only used for last vtarg, but we already zeroed it if last new = 1
    vpred = hcat(seg.vpred, seg.nextvpred)
    gaelam = zeros(Float32, N, T)
    rew = seg.rew
    lastgaelam = zeros(Float32, N)
    for t=T:-1:1
        nonterminal = 1-_news[:,t+1]
        δ = rew[:, t] + γ * vpred[:,t+1] .* nonterminal - vpred[:,t]
        gaelam[:, t] = lastgaelam = δ + γ * λ* nonterminal .* lastgaelam
    end
    seg.adv = gaelam
    seg.tdlamret = seg.adv .+ seg.vpred
    return seg
end




function PPOloss(clip, π, v, π_old, v_old, action, advantage, returns, o)
    """ Computes PPO loss
         Assumes discrete action space.
         Args:
             clip (float): probability ratio clipping range
             pi (Variable): discrete action logits, shaped [N x num_actions]
             v (Variable): value predictions, shaped [N x 1]
             pi_old (Variable): old discrete action logits, shaped [N x num_actions]
             v_old (Variable): old value predictions, shaped [N x 1]
             action (Variable): discrete actions, shaped [N x 1]
             advantage (Variable): action advantages, shaped [N x 1]
             returns (Variable): discounted returns, shaped [N x 1]
         Returns:
             policy_loss (Variable): policy surrogate loss, shaped [1]
             value_loss (Variable): value loss, shaped [1]
             entropy_loss (Variable): entropy loss, shaped [1]
    """
    action_prob = []
    action_prob_old = []
    prob = π_softmax(π)
    log_prob = logp(prob)  # slow according to pytorch

    prob_old = π_softmax(π_old)
    for i=1:length(action)
        push!( action_prob ,prob[action[i] + 1])
        push!(action_prob_old, prob_old[action[i]+1])
    end
         #action_prob_old = prob_old.gather(1, action)
    ratio = action_prob ./ (action_prob_old + Float32(1e-10))
    advantage = (advantage - mean(advantage)) / (std(advantage) + Float32(1e-5))
    surr1 = ratio .* advantage
    surr2 = zeros(Float32, length(surr1))
    for i=1:length(surr1)
        if ratio[i] > (1.+clip)
            surr2[i] = (1. + clip) * advantage[i]
        elseif ratio[i] < (1.-clip)
            surr2[i] = (1. - clip) * advantage[i]
        else
            surr2 = ratio[i] * advantage[i]
        end
    end
    policy_loss = -mean(min.(surr1, surr2))
    value_loss = mean(Float32(.5) * (v - returns).^2)
    entropy_loss = mean(sum(prob .* log_prob, 1))
    return policy_loss + value_loss * o["vfcoeff"] + entropy_loss * o["entcoeff"]
end


function the_loss(w, W_old, mb_obs, mb_actions, mb_advantages, mb_returns, o)

    mb_pis, mb_vs = predict(w, mb_obs)
    mb_pi_olds, mb_v_olds = predict(W_old, mb_obs)
    loss = PPOloss(o["clip"], mb_pis, reshape(mb_vs, 128), mb_pi_olds, mb_v_olds,
     mb_actions, mb_advantages, mb_returns, o)

     return loss
end


lossgradient = gradloss(the_loss) #loss.backward()





function play!(w, total_steps, venv, A, o)
    """ Runs PPO
     Args:
         total_steps (int): total number of environment steps to run for
     """
    N = o["num_actors"]
    T = o["worker_steps"]
    E = o["opt_epochs"]
    taken_steps = 0
    while taken_steps < total_steps
        progress = taken_steps / total_steps
        seg = interact(w, venv, A, o)
        seg = add_vtarg_and_adv(seg, o)
        # compute advantages, returns with GAE
        w_old = deepcopy(w)
        steps = N*T  # you may need to chnage  this
        for e = 1:E
            #@zero_grad policy

            #MB = steps // self.minibatch_steps

            b_obs = reshape(seg.ob[:,:,:,:,1:T], (84,84,4, steps)) #Variable(obs[:T].view((steps,) + ob_shape))
            b_rewards = reshape(seg.rew, (1, steps)) #Variable(rewards.view(steps, 1))
            b_masks = reshape(seg._new, (1, steps)) #Variable(masks.view(steps, 1))
            b_actions = reshape(seg.ac, (1, steps)) # Variable(actions.view(steps, 1))
            b_advantages = reshape(seg.adv, (1, steps)) #Variable(advantages.view(steps, 1))
            b_returns = reshape(seg.tdlamret, (1, steps)) #Variable(returns.view(steps, 1))

            b_inds = shuffle(collect(1:steps)) #np.arange(steps) np.random.shuffle(b_inds)


            for start = 1:o["minibatch-steps"]:steps
                mb_inds = b_inds[start:start + o["minibatch-steps"]-1]
                #mb_inds = b_inds

                mb_obs =  b_obs[:,:,:, mb_inds]
                mb_rewards =  b_rewards[mb_inds]
                mb_masks =  b_masks[mb_inds]
                mb_actions =  b_actions[mb_inds]
                mb_advantages =  b_advantages[mb_inds]
                mb_returns =  b_returns[mb_inds]

                #set_lr(self.optimizer, self.lr_func(progress))
                #self.optimizer.zero_grad()
                #torch.nn.utils.clip_grad_norm(self.policy.parameters(), self.max_grad_norm)
                g, mse = lossgradient(w, w_old, mb_obs, mb_actions, mb_advantages, mb_returns, o)
                update!(w, g) #self.optimizer.step()
            end
        taken_steps += steps
        println(taken_steps)
        end
    end
end
