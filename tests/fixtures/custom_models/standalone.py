"""Synthetic, non-biological builder using only built-in mechanisms."""
from neuron import h


def build_cell(size=20.0):
    soma = h.Section(name='synthetic_body')
    soma.L = soma.diam = size
    soma.insert('pas')
    soma.g_pas = 0.0001
    soma.e_pas = -70.0
    return soma
