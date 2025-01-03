import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def generate_graphical_inference():
    
    # Loading the generated scores dataset
    # Reading the final scores from the all three pipelines/modules
    df = pd.read_csv('../results/FinalScoringDataset.csv')
    
    x = np.arange(1, 31)
    sbert_score = df['sbert_score'].values
    simcse_score = df['simcse_score'].values
    hf_score = df['hf_score'].values
    plt.plot(x, sbert_score[0: 30], label="SBERT")
    plt.plot(x, simcse_score[0: 30], label = "SimCSE")
    plt.plot(x, hf_score[0: 30], label="Hugging Face", linestyle="--")
    plt.xlabel("Index Spanning")
    plt.ylabel("Similarity Scores")
    plt.title("Distribution of Similarity Scores")
    plt.legend()
    plt.show()
    plt.savefig('../results/SimilarityModule.png', dpi=300)
    plt.clf()
    camb_freq = [0] * 11
    spcy_freq = [0] * 11
    nltk_freq = [0] * 11
    # marks = np.arange(0, 11)
    camb_score = df["cam_score"]
    nltk_score = df["nltk_score"]
    spcy_score = df["spacy_score"]
    for i in camb_score:
        camb_freq[int(i)] += 1
    for i in nltk_score:
        nltk_freq[int(i)] += 1
    for i in spcy_score:
        spcy_freq[int(i)] += 1
    barWidth = 0.5
    br1 = np.arange(len(camb_freq))
    br2 = [x + barWidth for x in br1]
    br3 = [x + barWidth for x in br2]
    plt.bar(br1, nltk_freq, color='coral', label='NLTK Count')
    plt.bar(br2, spcy_freq, color='cyan', label='Spacy Count')
    plt.bar(br3, camb_freq, color='pink', label='Camembert Count')
    plt.xlabel("Score")
    plt.ylabel("Frequency of Occurance")
    plt.legend()
    plt.title("Frequency of Occurance of NER Scores")
    plt.show()
    plt.savefig('../results/NERModule.png', dpi=300)
    