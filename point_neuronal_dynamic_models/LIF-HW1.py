import numpy as np
import matplotlib.pyplot as plt


def calc_spike_frequency(spikes: list) -> float:
    if len(spikes) == 0:
        return 0.0
    if len(spikes) == 1:
        return 1 / spikes[0]
    return 1 / (spikes[-1] - spikes[-2])


# Model Parameters
sim_time = 50 * 1e-3                            # Simulation time           [seconds]            // 50 ms
dt = 0.1 * 1e-3                                 # Simulation time interval  [seconds]            // 0.1 ms
T = np.arange(0, sim_time + dt, dt)             # Time array                [list of seconds]
R_m = 1 * 1e3                                   # Membrane Resistance       [Ohm]                // 1 kOhm
tau_ref = 1 * 1e-3                              # Refractory Period         [seconds]            // 1 ms
V_rest = -70 * 1e-3                             # Resting potential         [Voltage]            // -70 mV
V_th = -40 * 1e-3                               # Spike threshold           [Voltage]            // -40 mV
V_spike = 50 * 1e-3                             # Spike voltage             [Voltage]            // 50 mV
dI = 1 * 1e-6                                   # Current intensity step    [Ampere]             // 1 uA
stimuli = np.arange(-100 * dI, 200 * dI, dI)    # Test stimuli for each tau [list of Amperes]

# Simulate for each tau
for C_m in [10 * 1e-6, 20 * 1e-6, 30 * 1e-6]:   # Testing different tau values by changing C_m, as C_m is used only for tau evaluation
    F = []                                      # Frequency list            [list of Hz]
    for I in stimuli:

        # Simulation parameters
        tau = R_m * C_m                         # Membrane time constant   [seconds]
        t_init = 0                              # Stimulus init time       [seconds]
        V_m = np.ones(len(T)) * V_rest          # Membrane voltage array   [list of Voltages]
        spikes = []                             # Spikes timings           [list of milliseconds]

        # Simulation
        for i, t in enumerate(T[:-1]):
            if t > t_init:
                V_m_inf_i = V_rest + R_m * I
                V_m[i + 1] = V_m_inf_i + (V_m[i] - V_m_inf_i) * np.exp(-dt / tau)
                if V_m[i] >= V_th:
                    # import ipdb; ipdb.set_trace(context=11)
                    spikes.append(t * 1e+3)
                    V_m[i] = V_spike
                    t_init = t + tau_ref

        # Calculate spike frequency
        a = calc_spike_frequency(spikes)        # As 1 divided time-cycle  [Hz]
        F.append(a)

    # Plot simulation results
    plt.figure(figsize=(10, 5))
    plt.title(f'Leaky Integrate-and-Fire Model (tau={tau:.2f})', fontsize=15)
    plt.xlabel('Stimulation Current Intensity (mA)', fontsize=15)
    plt.ylabel('Spike Frequency (Hz)', fontsize=15)
    plt.plot(stimuli * 1e+3, F, color='sandybrown', linewidth=2)
    plt.show()
    # plt.savefig(f"LIF_tau={tau:.2f}.png")
