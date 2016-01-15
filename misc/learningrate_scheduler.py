import numpy as np


class LearningRateScheduler(object):
    def get_weight_update(self, weight, delta_w, iteration_id):
        raise Exception('Not implemented')


class DecayingLearningRateWithMomentum(LearningRateScheduler):
    def __init__(self, base_learningrate=0.01, learningrate_decay_mode='none', learningrate_decay_half_life=1000,
                 initial_momentum=0.9, final_momentum=0.99, momentum_change_steps=1000, l2_decay=0.01):
        self.base_learningrate = base_learningrate
        self.learningrate_decay_mode = learningrate_decay_mode
        self.learningrate_decay_half_life = learningrate_decay_half_life
        self.initial_momentum = initial_momentum
        self.final_momentum = final_momentum
        self.momentum_change_steps = momentum_change_steps
        self.l2_decay = l2_decay
        self.current_iteration = 0
        assert self.learningrate_decay_mode in ['inverse-t', 'exponential', 'none']

    def set_iteration(self, iteration):
        self.current_iteration = iteration

    def get_learningrate_and_momentum(self, iteration_id):
        """
        if h.momentum_change_steps > step:
          f = float(step) / h.momentum_change_steps
          momentum = (1.0 - f) * h.initial_momentum + f * h.final_momentum
        else:
          momentum = h.final_momentum
        """
        momentum = self.final_momentum - (self.final_momentum - self.initial_momentum) * np.exp(-float(iteration_id)/self.momentum_change_steps)
        epsilon = self.base_learningrate
        if self.learningrate_decay_mode == 'inverse-t':
            epsilon = self.base_learningrate / (1 + float(iteration_id) / self.learningrate_decay_half_life)
        elif self.learningrate_decay_mode == 'exponential':
            epsilon = self.base_learningrate / np.power(2, float(iteration_id) / self.learningrate_decay_half_life)
        return momentum, epsilon

    def get_weight_update(self, weight, delta_w, old_delta_w=None):
        assert weight.shape == delta_w.shape
        momentum, epsilon = self.get_learningrate_and_momentum(self.current_iteration)
        if old_delta_w is not None:
            assert delta_w.shape == old_delta_w.shape
            dw = momentum * old_delta_w
        else:
            dw = np.zeros(delta_w.shape)

        if self.l2_decay is not None:
            dw += self.l2_decay * weight
        dw += delta_w
        dw *= epsilon
        return dw