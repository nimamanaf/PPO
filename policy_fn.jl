

atype = gpu()>=0 ? KnetArray{Float32} : Array{Float32}
w0(d...) = atype(xavier(Float32,d...))
b0(d...) = atype(zeros(Float32,d...))

function policy_fn(W, ob_space, ac_space): #pylint: disable=W0613
    #@assert isa(ob_space, gym.spaces.Box)
    x = ob_space / 255.0
    s = [i for i in size(x)]
    push!(s, 1)
    x = reshape(x, Tuple(s))
    x = convert(atype, x)
    w = W[1]
    w_a = W[2]
    w_̂v = W[3]

    π, V= π_net(w, w_a, w_̂v, x; pdrop=(0,0,0), actf1=relu, actf2=relu)

    return π, V
end

function π_init(ob_space, ac_space)
    W = Any[]
    num_actions = length(ac_space)
    h = (size(ob_space[1]), (5,5,3), 32)
    w = winit(h)
    w_a = Array{Any}(2)
    w_a[1] = w0(num_actions, size(w[end], 1))
    w_a[2] = b0(num_actions, 1)
    w_̂v = Array{Any}(2)
    w_̂v[1] = w0(1, size(w[end], 1))
    w_̂v[2] = b0(1, 1)
    push!(W, w)
    push!(W, w_a)
    push!(W, w_̂v)
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
            push!(w, xavier(w1, w2, cx, cy))
            push!(w, zeros(1, 1, cy, 1))
            x = (div(x1-w1+1, 2), div(x2-w2+1, 2), cy)
            # assuming conv4 with p=0, s=1 and pool with p=0,w=s=2
        elseif isa(h[i], Integer)
            push!(w, xavier(h[i], prod(x)))
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
"""

"""
DQN Natrure
x = tf.nn.relu(U.conv2d(x, 32, "l1", [8, 8], [4, 4], pad="VALID"))
x = tf.nn.relu(U.conv2d(x, 64, "l2", [4, 4], [2, 2], pad="VALID"))
x = tf.nn.relu(U.conv2d(x, 64, "l3", [3, 3], [1, 1], pad="VALID"))
x = U.flattenallbut0(x)
x = tf.nn.relu(tf.layers.dense(x, 512, name='lin', kernel_initializer=U.normc_initializer(1.0)))
"""
