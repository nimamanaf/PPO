
atype = Knet.gpu()>=0 ? KnetArray{Float32} : Array{Float32}
#w0(d...) = atype(.01*xavier(Float32,d...))
#b0(d...) = atype(zeros(Float32,d...))

function init_weights(inp, ha, hc, num_actions, atype)
    w0(d...) = atype(.01*randn(Float32,d...))
    b0(d...) = atype(zeros(Float32,d...))
    w = Dict()
    x = inp
    if length(x)>1
        xa =inp
        for i=1:length(ha)
            if isa(ha[i], Tuple)
                (x1,x2,cx) = xa
                (w1,w2,cy) = ha[i]
                w["wa_$i"] = w0(w1, w2, cx, cy)
                w["ba_$i"] = b0(1, 1, cy, 1)
                xa = (div(x1-w1+1, 2), div(x2-w2+1, 2), cy)
            # assuming conv4 with p=0, s=1 and pool with p=0,w=s=2
            elseif isa(ha[i], Integer)
                w["wa_$i"] = w0(ha[i], prod(xa))
                w["ba_$i"] = b0(ha[i],1)
                xa = ha[i]
            else
                error("Unknown layer type: $(h[i])")
            end
        end
        xc = inp
        for i=1:length(hc)
            if isa(hc[i], Tuple)
                (x1,x2,cx) = xc
                (w1,w2,cy) = hc[i]
                w["wc_$i"] = w0(w1, w2, cx, cy)
                w["bc_$i"] = b0(1, 1, cy, 1)
                xc = (div(x1-w1+1, 2), div(x2-w2+1, 2), cy)
            # assuming conv4 with p=0, s=1 and pool with p=0,w=s=2
            elseif isa(hc[i], Integer)
                w["wc_$i"] =  w0(hc[i], prod(xc))
                w["bc_$i"] = b0(hc[i],1)
                xc = hc[i]
            else
                error("Unknown layer type: $(h[i])")
            end
        end
    else

        for i=1:length(ha)
            w["wa_$i"] = .01*randn(ha[i], x)
            w["ba_$i"] = zeros(ha[i])
            x = ha[i]
        end
        x = inp
        for i=1:length(hc)
            w["wc_$i"] = .01*randn(hc[i], x)
            w["bc_$i"] = zeros(hc[i])
            x = hc[i]
        end
    end
    w["actor_w"] = w0(num_actions, ha[end])
    w["actor_b"] = b0(num_actions, 1)

    w["critic_w"] = w0(1, hc[end])
    w["critic_b"] = b0(1, 1)

    for k in keys(w)
        w[k] = convert(atype, w[k])
    end

    return w
end


function init_weights_shared(inp, h, num_actions, atype)
    w = Dict()
    for i=1:length(h)
        w["w_$i"] = .01*xavier(h[i], inp)
        w["b_$i"] = zeros(h[i])
        inp = h[i]
    end

    w["actor_w"] = .01*xavier(num_actions, h[end])
    w["actor_b"] = zeros(num_actions, 1)

    w["critic_w"] = .01*xavier(1, h[end])
    w["critic_b"] = zeros(1, 1)

    for k in keys(w)
        w[k] = convert(atype, w[k])
    end

    return w
end


function predict(w, inp; pdrop=(0,0,0), actf1=relu, actf2=relu)
    xa = inp
    xc = inp
    nh = Int((length(w)-4)/4)
    for i=1:nh
        if ndims(w["wa_$i"]) == 4     # convolutional layer
            xa = dropout(xa, pdrop[i==1?1:2])
            xa = conv4(w["wa_$i"], xa) .+ w["ba_$i"]
            xa = pool(actf1.(xa))
        elseif ndims(w["wa_$i"]) == 2 # fully connected layer
            xa = dropout(xa, pdrop[i==1?1:3])
            xa = w["wa_$i"]*mat(xa) .+ w["ba_$i"]
            xa = actf2.(xa)
        else
            error("Unknown layer type:")
        end
    end

    for i=1:nh
        if ndims(w["wc_$i"]) == 4     # convolutional layer
            xc = dropout(xc, pdrop[i==1?1:2])
            xc = conv4(w["wc_$i"], xc) .+ w["bc_$i"]
            xc = pool(actf1.(xc))
        elseif ndims(w["wc_$i"]) == 2 # fully connected layer
            xc = dropout(xc, pdrop[i==1?1:3])
            xc = w["wc_$i"]*mat(xc) .+ w["bc_$i"]
            xc = actf2.(xc)
        else
            error("Unknown layer type:")
        end
    end

    π = w["actor_w"] * xa .+ w["actor_b"]
    v = w["critic_w"] * xc .+ w["critic_b"]
    return π, v
end




function policy_fn(W, x) #pylint: disable=W0613
    #@assert isa(ob_space, gym.spaces.Box)
    #s = [i for i in size(x[1])]
    #push!(s, 1)
    #x = reshape(x, Tuple(s))
    #x = convert(atype, x)
    A = zeros(Float32, 84,84,4, length(x))
    for i=1:length(x)
      A[:,:,:,i]= x[i]
     end
    w = W[1:end-4]
    w_a = W[end-3:end-2]
    w_̂v = W[end-1:end]

    π, V= π_net(w, w_a, w_̂v, A; pdrop=(0,0,0), actf1=relu, actf2=relu)

    return π, V
end

function π_init(h, num_actions) #The last two array of arrays are related to policy and value function approximators
    #W = Any[]
    w = winit(h)
    #w_a = Any[]
    #w_a = Array{Float32,2}
    inp_size = size(w[end], 1)
    push!(w, w0(num_actions, inp_size))
    push!(w, b0(num_actions, 1))
    #w_̂v = Any[]
    push!(w, w0(1, inp_size))
    push!(w, b0(1, 1))
    #push!(W, w)
    #push!(W, w_a)
    #push!(W, w_̂v)
    return w
end
# Weight initialization for multiple layers
# h[i] is an integer for a fully connected layer, a triple of integers for convolution filters
# Output is an array [w0,b0,w1,b1,...,wn,bn] where wi,bi is the weight matrix/tensor and bias vector for the i'th layer
# winit(x,h1,h2,...,hn,y) for n hidden layer model
function winit(h)
    w = Any[]
    x = h[1]
    for i=2:length(h)
        if isa(h[i], Tuple)
            (x1,x2,cx) = x
            (w1,w2,cy) = h[i]
            push!(w, .01*xavier(w1, w2, cx, cy))
            push!(w, zeros(1, 1, cy, 1))
            x = (div(x1-w1+1, 2), div(x2-w2+1, 2), cy)
            # assuming conv4 with p=0, s=1 and pool with p=0,w=s=2
        elseif isa(h[i], Integer)
            push!(w, .01*xavier(h[i], prod(x)))
            push!(w, zeros(h[i],1))
            x = h[i]
        else
            error("Unknown layer type: $(h[i])")
        end
    end
    return map(a -> convert(atype, a), w)
    #map(atype, w)
end;


# Shape of x is of the type for knet
function π_net(w, w_a, w_̂v, x; pdrop=(0,0,0), actf1=relu, actf2=relu) # pdrop[1]:input, pdrop[2]:conv, pdrop[3]:fc
    for i=1:2:length(w)
        if ndims(w[i]) == 4     # convolutional layer
            x = dropout(x, pdrop[i==1?1:2])
            x = conv4(w[i], x) .+ w[i+1]
            x = pool(actf1.(x))
        elseif ndims(w[i]) == 2 # fully connected layer
            x = dropout(x, pdrop[i==1?1:3])
            x = w[i]*mat(x) .+ w[i+1]
            if i < length(w)-1; x = actf2.(x); end
        else
            error("Unknown layer type:")
        end
    end
    π = w_a[1]*mat(x) .+ w_a[2]
    V = w_̂v[1]*mat(x) .+ w_̂v[2]
    return π, V
end;


# Shape of x is of the type for knet
function π_net_sep(w, w_a, w_̂v, x; pdrop=(0,0,0), actf1=relu, actf2=relu) # pdrop[1]:input, pdrop[2]:conv, pdrop[3]:fc
    for i=1:2:length(w)
        if ndims(w[i]) == 4     # convolutional layer
            x = dropout(x, pdrop[i==1?1:2])
            x = conv4(w[i], x) .+ w[i+1]
            x = pool(actf1.(x))
        elseif ndims(w[i]) == 2 # fully connected layer
            x = dropout(x, pdrop[i==1?1:3])
            x = w[i]*mat(x) .+ w[i+1]
            if i < length(w)-1; x = actf2.(x); end
        else
            error("Unknown layer type:")
        end
    end
    π = w_a[1]*mat(x) .+ w_a[2]
    V = w_̂v[1]*mat(x) .+ w_̂v[2]
    return π, V
end;


"""
A3C
The network used a convolutional layer with 16 filters of size 8×8 with stride 4,
followed by a convolutional layer with with 32 filters of size 4×4 with stride 2,
followed by a fully connected layer with 256 hidden units. All three hidden layers
were followed by a rectifier nonlinearity. The value-based methods had a single
linear output unit for each action representing the action-value. The model used
by actor-critic agents had two set of outputs – a softmax output with one entry
per action representing the probability of selecting the action, and a single linear
output representing the value function.
    x = tf.nn.relu(U.conv2d(x, 16, "l1", [8, 8], [4, 4], pad="VALID"))
    x = tf.nn.relu(U.conv2d(x, 32, "l2", [4, 4], [2, 2], pad="VALID"))
    x = U.flattenallbut0(x)
    x = tf.nn.relu(tf.layers.dense(x, 256, name='lin', kernel_initializer=U.normc_initializer(1.0)))

DQN Natrure
x = tf.nn.relu(U.conv2d(x, 32, "l1", [8, 8], [4, 4], pad="VALID"))
x = tf.nn.relu(U.conv2d(x, 64, "l2", [4, 4], [2, 2], pad="VALID"))
x = tf.nn.relu(U.conv2d(x, 64, "l3", [3, 3], [1, 1], pad="VALID"))
x = U.flattenallbut0(x)
x = tf.nn.relu(tf.layers.dense(x, 512, name='lin', kernel_initializer=U.normc_initializer(1.0)))
"""
