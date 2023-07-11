Contains low-value files that I don't want to delete in case they prove useful later.

### Custom Path Patching?

All use cases for context neuron at L3/5
- direct effect of MLP3
- total effect of MLP3
- indirect effect of MLP3 via MLP4 and MLP5
- indirect effect of MLP3 via MLP4 + direct effect of MLP3
- indirect effect of MLP3 via MLP5 + direct effect of MLP3
- indirect effect of MLP3 via MLP4 and indirect effect of MLP3 via MLP4 and MLP5
- indirect effect of MLP3 via 
- indirect effect of MLP3 via MLP4 only
- indirect effect of MLP3 via MLP5 only


Rephrase attempt of interesting use cases: We activate the German neuron and let its effect propagate through select paths to the output.
- direct effect: we let it propagate through the MLP3 -> output path
- total effect: we let it propagate through all paths
- indirect effect of MLP3 via MLP4 and MLP5: we let the German neuron propagate from MLP3 -> MLP4 -> output, MLP3 -> MLP5 -> output, and MLP3 -> MLP4 -> MLP5 -> output
- indirect effect of MLP3 via MLP5 only: we let the German neuron propagate from MLP3 -> MLP5 -> output (no MLP3 -> MLP4 -> MLP5 -> output)

These specifications could be described as lists of paths that the German neuron's value can propagate through. E.g. for a single run we could pass in an interface like:
[('MLP3', 'MLP4', 'MLP5', 'out'), ('MLP3', 'MLP4', 'out')]
If we want to get total effect we could also specify a shorthand for that:
[('MLP3', 'all')]

The implementation will doubtless be messier, but a clean interface will let us encapsulate these concerns and avoid being concerned that we misunderstand the situation.

In Callum's implementation the strings in the path definition tuples are replaced with Nodes with a "name" property corresponding to the TransformerLens component
naming convention, e.g. "blocks.3.mlp.hook_post", and also let you specify neurons and heads e.g. Node("blocks.3.mlp.hook_post", neuron=1337). The tuple itself is
replaced with an IterNode class.

### Existing TransformerLens Activation Path Methods?

Try generic_activation_patch with context neuron. It patches in one cache to one component. You can run in a with.hooks() context manager so get the run to be with hooks (e.g. with the German context neuron ablated, then you can pass in a cache with it enabled). There are helper functions to run it for multiple components. The limitations are:
- The patch for each component is run separately.
- A metric function is called, rather than returning the cache. Potentially we could pass in an identity function that just returns our cache.


### Notes

Example procedure for calculating direct effect
- Get a cache with the context neuron ablated
- Run with the context neuron active and patch in all activations after MLP3

Example procedure for calculating indirect effect
- Get a cache with the context neuron active
- Run with the context neuron ablated and patch in the next activations


