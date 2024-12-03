import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

with open('/Users/itsdc03/Desktop/Reinforcement Learning/pytorch_car_caring/data/train2.txt', "r") as file:
    data = file.readlines()

# Clean the data to remove parentheses and split into columns
cleaned_data = [line.strip().replace("(", "").replace(")", "").split(", ") for line in data]

train_df = pd.DataFrame(cleaned_data, columns=["episode", "score", "moving_average"])

# Convert columns to numeric types
train_df["episode"] = pd.to_numeric(train_df["episode"])
train_df["score"] = pd.to_numeric(train_df["score"])
train_df["moving_average"] = pd.to_numeric(train_df["moving_average"])



def clean_test(file_path):

    # Read the file and preprocess it
    with open(file_path, "r") as file:
        data = file.readlines()

    # Extract data for Episodes, Score, and Frames
    cleaned_data = []
    for line in data:
        parts = line.strip().split()
        episode = int(parts[1])  # Extract episode number
        score = float(parts[3].rstrip(","))  # Extract score, remove trailing comma
        frames = int(parts[5])  # Extract frame count
        cleaned_data.append([episode, score, frames])

    # Create the DataFrame
    df = pd.DataFrame(cleaned_data, columns=["Episode", "Score", "Frames"])

    # Display the DataFrame
    samples = df.sample(n=1000, replace=True, weights=None, random_state=42)

    # Assign new sequential Episode numbers
    samples = samples.reset_index(drop=True)
    samples['Episode'] = np.arange(1000)

    # Rearrange columns for consistency
    samples = samples[['Episode', 'Score', 'Frames']]

    # Display the first few rows
    return samples

eval_df = clean_test('/Users/itsdc03/Desktop/Reinforcement Learning/pytorch_car_caring/data/test_data_sb_model.csv')
test_df = clean_test('/Users/itsdc03/Desktop/Reinforcement Learning/pytorch_car_caring/data/test_data_self_model.csv')

def plot_line(x, y, x_name, y_name, type = 'Train'):
    _, ax = plt.subplots()
    ax.plot(x, y)
    ax.set_title(y_name)
    ax.set_xlabel(x_name)
    ax.set_ylabel(y_name)

    plt.savefig(f'/Users/itsdc03/Desktop/Reinforcement Learning/pytorch_car_caring/plots/line/{type}_{y_name} line.png')
    plt.show()

def plot_hist(y, y_name, type = 'Test', bins = [(i-1)*100 for i in range(12)]):
    _, ax = plt.subplots()
    ax.hist(y, bins = bins)
    ax.set_xlabel(y_name)
    ax.set_title(y_name)
    ax.set_ylabel('Episodes')

    plt.savefig(f'/Users/itsdc03/Desktop/Reinforcement Learning/pytorch_car_caring/plots/hist/{type}_{y_name} hist.png')
    plt.show()



plot_line(train_df['episode'], train_df['score'], 'episode', 'score', 'train')
plot_line(train_df['episode'], train_df['moving_average'], 'episode', 'moving average', 'train')

plot_hist(test_df['Frames'], 'Frames Used (self-model)', 'test')
plot_hist(test_df['Score'], 'Self-model Score', 'test')

plot_hist(eval_df['Frames'], 'Frames Used (StableBaselines-model)', 'eval')
plot_hist(eval_df['Score'], 'StableBaselines-model Score', 'eval')

