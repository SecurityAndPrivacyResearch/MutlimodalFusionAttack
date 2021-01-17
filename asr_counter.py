import os

for _, _, files in os.walk('adversarial_samples'):
  break

trues = 0.
total = 0.

for f in files:
  if '.pt' in f:
    if 'True' in f:
      trues += 1
    total += 1

print('% True:', (trues / total) * 100)
