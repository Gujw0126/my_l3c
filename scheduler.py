class LRSchedule(object):
    def __init__(self, optims):
        self.optims = optims
        self._current_lr = None

    def _get_lr(self, i):
        raise NotImplementedError()

    def update(self, i):
        lr = self._get_lr(i)
        if lr == self._current_lr:
            return
        for optim in self.optims:
            for pg in optim.param_groups:
                pg['lr'] = lr
        self._current_lr = lr


class ConstantLRSchedule(object):
    def update(self, i):  # no-op
        pass



class ExponentialDecayLRSchedule(LRSchedule):
    def __init__(self, optims, initial, decay_fac,
                 decay_interval_itr=None, decay_interval_epoch=None, epoch_len=None,
                 warm_restart=None,
                 warm_restart_schedule=None):
        super(ExponentialDecayLRSchedule, self).__init__(optims)
        #assert_exc((decay_interval_itr is not None) ^ (decay_interval_epoch is not None), 'Need either iter or epoch')
        if decay_interval_epoch:
            assert epoch_len is not None
            decay_interval_itr = int(decay_interval_epoch * epoch_len)
            if warm_restart:
                warm_restart = int(warm_restart * epoch_len)
        self.initial = initial
        self.decay_fac = decay_fac
        self.decay_every_itr = decay_interval_itr

        self.warm_restart_itr = warm_restart
        self.warm_restart_schedule = warm_restart_schedule

        self.last_warm_restart = 0

    def _get_lr(self, i):
        if i > 0 and self.warm_restart_itr and ((i - self.last_warm_restart) % self.warm_restart_itr) == 0:
            if i != self.last_warm_restart:
                self._warm_restart()
                self.last_warm_restart = i
        i -= self.last_warm_restart
        num_decays = i // self.decay_every_itr
        return self.initial * (self.decay_fac ** num_decays)

    def _warm_restart(self):
        print('WARM restart')
        if self.warm_restart_schedule:
            self.initial = self.warm_restart_schedule.initial
            self.decay_fac = self.warm_restart_schedule.decay_fac
            self.decay_every_itr = self.warm_restart_schedule.decay_every_itr
            self.warm_restart_itr = self.warm_restart_schedule.warm_restart_itr
            self.warm_restart_schedule = self.warm_restart_schedule.warm_restart_schedule