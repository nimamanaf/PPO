##############################################################################
# Defines s PPO type with constructor
##############################################################################
type PPOObjective
    """ Computes PPO objectives
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
    clip::Float32 #probability ratio clipping range
    π::Array{Float32}  #discrete action logits, shaped [N x num_actions]
    v::Vector{Float32} #value predictions, shaped [N x 1]
    π_old::Array{Float32}  #old discrete action logits, shaped [N x num_actions]
    v_old::Vector{Float32} #value predictions, shaped [N x 1]
    action::Vector{Int32} #discrete actions, shaped [N x 1]
    advantage::Vector{Float32} #action advantages, shaped [N x 1]
    returns::Vector{Float32} #discounted returns, shaped [N x 1]

    addContent::Function
    setHeader::Function
    getHeaders::Function
    getContents::Function
    getResponse::Function

    function Response()
        self = new()

        self... = function ()

        end


        return this
    end
end

class PPOObjective(nn.Module):
    def forward(self, clip, pi, v, pi_old, v_old, action, advantage, returns):
        """ Computes PPO objectives
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
        prob = Fnn.softmax(pi)
        log_prob = Fnn.log_softmax(pi)
        action_prob = prob.gather(1, action)

        prob_old = Fnn.softmax(pi_old)
        action_prob_old = prob_old.gather(1, action)

        ratio = action_prob / (action_prob_old + 1e-10)

        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-5)

        surr1 = ratio * advantage
        surr2 = torch.clamp(ratio, min=1. - clip, max=1. + clip) * advantage

        policy_loss = -torch.min(surr1, surr2).mean()
        value_loss = (.5 * (v - returns) ** 2.).mean()
        entropy_loss = (prob * log_prob).sum(1).mean()

return policy_loss, value_loss, entropy_loss





struct PPO
    π::Array{Float32}  #discrete action logits, shaped [N x num_actions]
    venv::GymEnv
    test_env::GymEnv
    optimizer
    lr_func::Function
    clip_func::Function
    γ=::Float32
    λ=::Float32
    worker_steps::Int32
    sequence_steps::Int32
    minibatch_steps::Int32
    opt_epochs::Int32
    value_coef::Float32
    entropy_coef::Float32
    max_grad_norm::Float32
    cude::Bool
    plot_reward::Bool
    plot_points::Int32
    plot_path::String
    test_repeat_max::Int32

    PPO( policy, venv, test_env, optimizer) = new( policy, venv, test_env,
     optimizer, lr_func=None, clip_func=None, gamma=.99, lambd=.95,
                 worker_steps=128, sequence_steps=32, minibatch_steps=256,
                 opt_epochs=3, value_coef=1., entropy_coef=.01, max_grad_norm=.5,
                 cuda=False, plot_reward=False, plot_points=20, plot_path='ep_reward.png',
test_repeat_max=100))

      """ Proximal Policy Optimization algorithm class
      Evaluates a policy over a vectorized environment and
      optimizes over policy, value, entropy objectives.
      Assumes discrete action space.
      Args:
          policy (nn.Module): the policy to optimize
          venv (vec_env): the vectorized environment to use
          test_env (Env): the environment to use for policy testing
          optimizer (optim.Optimizer): the optimizer to use
          clip (float): probability ratio clipping range
          gamma (float): discount factor
          lambd (float): GAE lambda parameter
          worker_steps (int): steps per worker between optimization rounds
          sequence_steps (int): steps per sequence (for backprop through time)
          batch_steps (int): steps per sequence (for backprop through time)
      """
      self.policy = policy
      self.policy_old = copy.deepcopy(policy)
      self.venv = venv
      self.test_env = test_env
      self.optimizer = optimizer

      self.lr_func = lr_func
      self.clip_func = clip_func

      self.num_workers = venv.num_envs
      self.worker_steps = worker_steps
      self.sequence_steps = sequence_steps
      self.minibatch_steps = minibatch_steps

      self.opt_epochs = opt_epochs
      self.gamma = gamma
      self.lambd = lambd
      self.value_coef = value_coef
      self.entropy_coef = entropy_coef
      self.max_grad_norm = max_grad_norm
      self.cuda = cuda

      self.plot_reward = plot_reward
      self.plot_points = plot_points
      self.plot_path = plot_path
      self.ep_reward = np.zeros(self.num_workers)
      self.reward_histr = []
      self.steps_histr = []

      self.objective = PPOObjective()

      self.last_ob = self.venv.reset()

      self.taken_steps = 0

self.test_repeat_max = test_repeat_max
