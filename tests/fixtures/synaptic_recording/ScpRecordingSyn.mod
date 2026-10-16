TITLE Original synthetic synapse for SCP recording contract tests

COMMENT
This small test-only mechanism is original SCP fixture code. It is not a
scientific cell or fitted synapse model and contains no third-party source.
ENDCOMMENT

NEURON {
    POINT_PROCESS ScpRecordingSyn
    NONSPECIFIC_CURRENT i
    RANGE initW, tau, e, i, g, i_AMPA, i_NMDA, g_AMPA, g_NMDA
    RANGE record_use, record_Pr
}

UNITS {
    (nA) = (nanoamp)
    (mV) = (millivolt)
    (uS) = (microsiemens)
}

PARAMETER {
    initW = 0.002 (uS)
    tau = 2 (ms)
    e = 0 (mV)
}

ASSIGNED {
    v (mV)
    i (nA)
    g (uS)
    i_AMPA (nA)
    i_NMDA (nA)
    record_use
    record_Pr
}

STATE {
    g_AMPA (uS)
    g_NMDA (uS)
}

INITIAL {
    g_AMPA = 0
    g_NMDA = 0
    record_use = 0.25
    record_Pr = 0.5
}

BREAKPOINT {
    SOLVE decay METHOD cnexp
    g = g_AMPA + g_NMDA
    i_AMPA = g_AMPA * (v - e)
    i_NMDA = g_NMDA * (v - e)
    i = i_AMPA + i_NMDA
}

DERIVATIVE decay {
    g_AMPA' = -g_AMPA / tau
    g_NMDA' = -g_NMDA / (2 * tau)
}

NET_RECEIVE (weight) {
    g_AMPA = g_AMPA + initW * weight
    g_NMDA = g_NMDA + 0.5 * initW * weight
    record_use = record_use + 0.05
    record_Pr = record_Pr + 0.01
}
