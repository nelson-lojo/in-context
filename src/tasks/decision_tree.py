from task import *

class DecisionTree(Task):
    def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None, depth=4):

        super(DecisionTree, self).__init__(n_dims, batch_size, pool_dict, seeds)
        self.depth = depth

        if pool_dict is None:

            # We represent the tree using an array (tensor). Root node is at index 0, its 2 children at index 1 and 2...
            # dt_tensor stores the coordinate used at each node of the decision tree.
            # Only indices corresponding to non-leaf nodes are relevant
            self.dt_tensor = torch.randint(
                low=0, high=n_dims, size=(batch_size, 2 ** (depth + 1) - 1)
            )

            # Target value at the leaf nodes.
            # Only indices corresponding to leaf nodes are relevant.
            self.target_tensor = torch.randn(self.dt_tensor.shape)
        elif seeds is not None:
            self.dt_tensor = torch.zeros(batch_size, 2 ** (depth + 1) - 1)
            self.target_tensor = torch.zeros_like(self.dt_tensor)
            generator = torch.Generator()
            assert len(seeds) == self.b_size
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                self.dt_tensor[i] = torch.randint(
                    low=0,
                    high=n_dims - 1,
                    size=2 ** (depth + 1) - 1,
                    generator=generator,
                )
                self.target_tensor[i] = torch.randn(
                    self.dt_tensor[i].shape, generator=generator
                )
        else:
            raise NotImplementedError

    def evaluate(self, xs_b):
        dt_tensor = self.dt_tensor.to(xs_b.device)
        target_tensor = self.target_tensor.to(xs_b.device)
        ys_b = torch.zeros(xs_b.shape[0], xs_b.shape[1], device=xs_b.device)
        for i in range(xs_b.shape[0]):
            xs_bool = xs_b[i] > 0
            # If a single decision tree present, use it for all the xs in the batch.
            if self.b_size == 1:
                dt = dt_tensor[0]
                target = target_tensor[0]
            else:
                dt = dt_tensor[i]
                target = target_tensor[i]

            cur_nodes = torch.zeros(xs_b.shape[1], device=xs_b.device).long()
            for j in range(self.depth):
                cur_coords = dt[cur_nodes]
                cur_decisions = xs_bool[torch.arange(xs_bool.shape[0]), cur_coords]
                cur_nodes = 2 * cur_nodes + 1 + cur_decisions

            ys_b[i] = target[cur_nodes]

        return ys_b

    @staticmethod
    def generate_pool_dict(n_dims, num_tasks, hidden_layer_size=4, **kwargs):
        raise NotImplementedError

    @staticmethod
    def get_metric():
        return squared_error

    @staticmethod
    def get_training_metric():
        return mean_squared_error