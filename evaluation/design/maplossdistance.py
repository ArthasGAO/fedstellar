import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker

MIN_LOSS = 1e-6

def maplossdist(local_loss, nei_loss, tau):
    k = 1 / max(MIN_LOSS, local_loss)
    loss_dist = max(nei_loss - local_loss, 0)
    w = math.exp(-k * loss_dist)
    # if (w < tau) or (math.isnan(w)):
    #    return float(0)
    return w

x = np.linspace(0, 10, 100)
loss_values = [0.5, 1.0, 2, 5]  # Add more values as needed
tau_values = [0.1, 0.3, 0.5]  # Add more values as needed

line_styles = ['-', '--', '-.', ':', '-']

for i in range(len(loss_values)):
    #for tau in tau_values:
    loss = loss_values[i]

    y = [maplossdist(loss, x_val, 0.5) for x_val in x]

    style=line_styles[i % len(line_styles)]
    label = f"\u03C6(lᵢ)={loss}"
    plt.plot(x, y, label=label,linewidth=1, linestyle=style)
    plt.xticks(np.arange(min(x), max(x)+1, 1.0))
    #plt.yticks(np.arange(0, 1.1, 0.1))


tau = 0.5
# y_tau = [tau for x_val in x]
# plt.plot(x, y_tau, label=label,linewidth=1, )
plt.xlabel('Neighbor Loss')
plt.ylabel('w')
plt.ylim(-0.05, 1.05)
plt.xlim(-0.05, 10)
plt.legend()
# plt.title('Loss Functions')
plt.legend(fontsize=10)
# Set grid
plt.grid(True, linestyle='--', alpha=0.5)
# Increase tick font sizes
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
# plt.show()

# plt.savefig('loss_mapping.svg')
plt.savefig('loss_mapping.svg')