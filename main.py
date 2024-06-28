import subprocess

# Run program1.py to define and compile the model
print("Running program1.py...")
subprocess.run(['python', 'program1.py'])

# Run program2.py to generate noisy data and train the model
print("Running program2.py...")
subprocess.run(['python', 'program2.py'])

# Run program3.py to preprocess, denoise, and save images
print("Running program3.py...")
subprocess.run(['python', 'program3.py'])
