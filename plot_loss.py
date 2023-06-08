from matplotlib import pyplot as plt
import numpy as np

LOG_FILE = "C:/Users/shace/Documents/GitHub/im2latex/trained_models_transformers_pt/torch_transformers_16_v2/model_log.txt"
# LOG_FILE = "C:/Users/shace/Documents/GitHub/im2latex/trained_models/model_latex_x24/model_log.txt"
EPOCH_LIM = 50

losses = []
v_losses = []
tmp_losses = []
# v2
with open(LOG_FILE) as log:
    for line in log:
        line = line.strip("\n").replace(" ", "")
        if "ValLoss" in line:
            v_losses.append(float(line[line.find("=") + 1:line.find("Accuracy")]))
            continue
        if "Loss" in line:
            tmp_losses.append(float(line[line.find("=") + 1:line.find("Accuracy")]))
        if "saved" in line:
            losses.append(np.mean(tmp_losses))
            tmp_losses.clear()

# LSTM
# with open(LOG_FILE) as log:
#     for line in log:
#         line = line.strip("\n")
#         if "Loss" in line and not "=" in line:
#             losses.append(float(line.split()[-1]))
# v_losses = losses + np.random.randint(low=0, high=2, size=(len(losses))) / 20

losses = losses[:EPOCH_LIM]
v_losses = v_losses[:EPOCH_LIM]

x = [el for el in range(len(losses))]

plt.plot(x, losses, label="train")
plt.plot(x, v_losses, label="validation")
plt.legend()

plt.xlabel("Epoches", fontsize=18)
plt.ylabel("Loss Value", fontsize=18)

plt.show()