This project builds a neural network model that predicts the next pitch type thrown by MLB pitcher Yoshinobu Yamamoto, who uses a diverse six-pitch mix:

FF – Four-Seam Fastball

FS – Splitter

SL – Slider

CU – Curveball

FC – Cutter

SI – Sinker

Pitch prediction is framed as a sequence classification problem, where each plate appearance (PA) is treated as a variable-length sequence of pitches. At every timestep, the model uses player-specific and contextual statistics to identify the most likely next pitch.

The final system achieves approximately 82% accuracy on a leak-proof, game-level validation split.
