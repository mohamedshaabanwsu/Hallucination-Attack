import pandas as pd
import matplotlib.pyplot as plt

# Load Excel file
# Change "yourfile.xlsx" to your actual filename
df = pd.read_excel("run_log_instruct_#5.xlsx")

# Suppose your Excel has two columns: "Steps" and "Loss"
steps = df["Step"]
loss = df["Temp Loss"]

# Plot
plt.plot(steps, loss, color="darkred")

# Labels
plt.xlabel("Steps", fontsize=14, fontweight="bold", color="darkred")
plt.ylabel("Loss", fontsize=14, fontweight="bold", color="darkred")

# Grid and ticks
plt.grid(True)
plt.xticks(range(4, int(max(steps))+1, 4))  # x-axis increments of 4

plt.show()
