from neuron import h


class SyntheticCell:
    def __init__(self, length, use_mod=False):
        self.body = h.Section(name='body', cell=self)
        self.branch = h.Section(name='branch', cell=self)
        self.body.L = self.body.diam = length
        self.branch.L, self.branch.diam, self.branch.nseg = 60.0, 2.0, 3
        self.branch.connect(self.body(1.0))
        self.whole_cell = [self.body, self.branch]
        for section in self.whole_cell:
            section.insert('pas')
            section.g_pas, section.e_pas = 0.0001, -70.0
            if use_mod:
                section.insert('scp_synthetic_leak')
