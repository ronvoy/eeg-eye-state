import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset (chronological order)
df = pd.read_csv('eeg_eye_state.csv')

# Plotting eyeDetection (0 and 1) counts over time
plt.figure(figsize=(14, 6))
plt.plot(df['eyeDetection'].rolling(window=50, min_periods=1).mean(), label='Rolling Mean (window=50)')
plt.scatter(df.index, df['eyeDetection'], s=2, alpha=0.3, label='Raw eyeDetection')
plt.title('EEG Eye State: eyeDetection (0=Closed, 1=Open) Over Time')
plt.xlabel('Time (Sample Index)')
plt.ylabel('eyeDetection')
plt.yticks([0, 1], ['Closed', 'Open'])
plt.legend()
plt.tight_layout()
plt.show()