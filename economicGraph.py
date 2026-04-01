import numpy as np
import matplotlib.pyplot as plt

AI = np.linspace(0, 1, 100)

automation_effect = -0.4 * AI 
productivity_effect = 0.25 * AI 

net_effect = automation_effect + productivity_effect

plt.plot(AI, automation_effect, label="Automation effect")
plt.plot(AI, productivity_effect, label="Productivity effect")
plt.plot(AI, net_effect, label="Net employment effect")

plt.xlabel("AI Adoption Level")
plt.ylabel("Employment Change")
plt.legend()

plt.show()