An implementation of a Compositional Pattern Producing Network based off of the one from blog.otoro.net.
Implemented in Pytorch and it comes with some fancy features for specifying what the network can output.

# The Network Structure
There are two ways to set the network structure. First is with a specification, which takes the number of nodes
for each layer as the first argument, and the number of layers as the second argument. So if you were to pass
in 16 4, it would create 4 hidden linear layers each with 16 nodes.
For example you would run cppn.py --spec "16 4"
The other way to create a net is with a configuration that consist of a list of numbers that represent the number
of nodes in sequential linear layers.
For example cppn.py --config "256 128 64 32 16 8 4" will produce nice results.

# Output format
You can specify whether you want it to be video format or picture format with the -v and -p flags.
For example cppn.py -v true