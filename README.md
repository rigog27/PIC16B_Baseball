This project builds a recurrent neural network that predicts the next pitch type thrown by MLB pitcher Yoshinobu Yamamoto, who uses a diverse six-pitch mix:

FF – Four-Seam Fastball

FS – Splitter

SL – Slider

CU – Curveball

FC – Cutter

SI – Sinker

Pitch prediction is framed as a sequence classification problem, where each plate appearance (PA) is treated as a variable-length sequence of pitches. At every timestep, the model uses player-specific and contextual statistics to identify the most likely next pitch.

The final system achieves approximately 82% accuracy on a leak-proof, game-level validation split.

Library Requirements:
pybaseball, Numpy, Pytorch, Pandas, matplotlib

To run, execute lines !train.py and !evaluate.py through the main.ipynb Jupyter Notebook. Requires all files contained inside the Final Version Running Code Folder. Both evaluate and train require model.py to run as well. 
