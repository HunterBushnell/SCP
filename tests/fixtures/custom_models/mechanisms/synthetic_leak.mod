TITLE Synthetic non-biological integration fixture
NEURON {
    SUFFIX scp_synthetic_leak
    NONSPECIFIC_CURRENT i
    RANGE g, e, i
}
UNITS {
    (mV) = (millivolt)
    (mA) = (milliamp)
    (S) = (siemens)
}
INCLUDE "leak_parameters.inc"
ASSIGNED {
    v (mV)
    i (mA/cm2)
}
BREAKPOINT { i = g * (v - e) }
