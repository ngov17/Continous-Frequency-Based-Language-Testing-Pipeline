import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import pandas as pd
import nltk
from nltk.corpus import brown
import random

# ------------------------------
# 1. Prepare Lexicon
# ------------------------------
nltk.download('brown')
nltk.download('universal_tagset')

words = list(set(brown.words()))
words = [w.lower() for w in words if w.isalpha()]
words = words[:5000]

tagged = brown.tagged_words(tagset='universal')
pos_map = {w.lower(): t for w, t in tagged}

def assign_contour(pos_tag):
    if pos_tag in ['NOUN', 'PRON']:
        return "rise+level", np.random.randint(300, 500)
    elif pos_tag in ['VERB']:
        return "level", np.random.randint(350, 450)
    elif pos_tag in ['ADJ']:
        return "slow-rise", np.random.randint(400, 550)
    elif pos_tag in ['ADV']:
        return "terminal-rise", np.random.randint(450, 600)
    elif pos_tag in ['DET', 'ADP']:
        return "sustain", np.random.randint(250, 350)
    else:
        return "fall-level", np.random.randint(200, 400)

lexicon = {}
for w in words:
    tag = pos_map.get(w, 'NOUN')
    contour, freq = assign_contour(tag)
    lexicon[w] = {"freq": freq, "dur": 0.2, "contour": contour, "pos": tag}

# ------------------------------
# 2. Generate CFBL Corpus
# ------------------------------
def generate_utterance(max_len=8):
    length = np.random.randint(3, max_len+1)
    return random.sample(words, length)

corpus = [generate_utterance() for _ in range(500)]

# ------------------------------
# 3. Synthesize Signals
# ------------------------------
def contour_to_signal(entry, sr=22050):
    freq = entry["freq"]
    dur = entry["dur"]
    t = np.linspace(0, dur, int(sr*dur))
    contour = entry["contour"]
    if contour in ["rise+level", "slow-rise"]:
        signal = np.sin(2*np.pi*(freq + np.linspace(0, 50, t.size))*t)
    elif contour in ["fall-level", "slow-fall"]:
        signal = np.sin(2*np.pi*(freq - np.linspace(0, 50, t.size))*t)
    elif contour == "terminal-rise":
        signal = np.sin(2*np.pi*(freq + np.sin(np.linspace(0, np.pi, t.size))*50)*t)
    elif contour == "sustain":
        signal = np.sin(2*np.pi*freq*t)
    else:
        signal = np.sin(2*np.pi*freq*t)
    return signal

def utterance_to_signal(utterance, sr=22050):
    signals = [contour_to_signal(lexicon[word], sr) for word in utterance]
    return np.concatenate(signals)

# ------------------------------
# 4. Extract Features
# ------------------------------
def extract_features(signal, sr=22050, frame_size=2048, hop_length=512):
    pitches, magnitudes = librosa.piptrack(y=signal, sr=sr, n_fft=frame_size, hop_length=hop_length)
    freqs = [np.mean(pitches[:, i][pitches[:, i] > 0]) if np.any(pitches[:, i] > 0) else 0 for i in range(pitches.shape[1])]
    freqs = np.array(freqs)
    freqs = freqs[freqs > 0]
    if len(freqs) == 0:
        return {}
    return {
        "mean_freq": np.mean(freqs),
        "max_freq": np.max(freqs),
        "min_freq": np.min(freqs),
        "freq_slope": np.mean(np.diff(freqs)),
        "amplitude_mean": np.mean(np.abs(signal)),
        "duration": len(signal)/sr
    }

features_list = [extract_features(utterance_to_signal(u)) for u in corpus]
features_df = pd.DataFrame(features_list)

# ------------------------------
# 5. Mock Dependency Parsing
# ------------------------------
def parse_dependency(utterance):
    if len(utterance) == 0:
        return {}
    head_idx = np.random.randint(len(utterance))
    deps = {}
    for i, word in enumerate(utterance):
        if i != head_idx:
            deps[word] = utterance[head_idx]
    return {"head": utterance[head_idx], "dependencies": deps}

parsed_corpus = [parse_dependency(u) for u in corpus]

# ------------------------------
# 6. Compute Metrics
# ------------------------------
all_words = [w for u in corpus for w in u]
unique_words = set(all_words)
ttr = len(unique_words)/len(all_words)

num_dependents = [len(p["dependencies"]) for p in parsed_corpus]

metrics = {
    "num_utterances": len(corpus),
    "avg_utterance_length": np.mean([len(u) for u in corpus]),
    "avg_segment_duration": np.mean(features_df["duration"]),
    "mean_freq_overall": np.mean(features_df["mean_freq"]),
    "max_freq_overall": np.max(features_df["max_freq"]),
    "min_freq_overall": np.min(features_df["min_freq"]),
    "dependency_coverage": np.mean([len(p["dependencies"])/len(u) if len(u)>1 else 1 for p,u in zip(parsed_corpus, corpus)]),
    "lexical_richness_TTR": ttr
}
metrics_df = pd.DataFrame([metrics])
print(metrics_df)

# ------------------------------
# 7. Visualization: Combined Figure
# ------------------------------
fig, axs = plt.subplots(2,2, figsize=(14,10))

# 7a. Mean Frequency Distribution
axs[0,0].hist(features_df["mean_freq"], bins=30, color='skyblue', edgecolor='black')
axs[0,0].set_title("Distribution of Mean Frequencies")
axs[0,0].set_xlabel("Mean Frequency (Hz)")
axs[0,0].set_ylabel("Utterances")

# 7b. Segment Duration Distribution
axs[0,1].hist(features_df["duration"], bins=30, color='salmon', edgecolor='black')
axs[0,1].set_title("Distribution of Segment Durations")
axs[0,1].set_xlabel("Duration (s)")
axs[0,1].set_ylabel("Utterances")

# 7c. Number of Dependents per Head (Syntactic Richness)
axs[1,0].hist(num_dependents, bins=range(0, max(num_dependents)+2), color='lightgreen', edgecolor='black', align='left')
axs[1,0].set_title("Syntactic Richness: Dependents per Head")
axs[1,0].set_xlabel("Number of Dependents")
axs[1,0].set_ylabel("Utterances")

# 7d. Lexical vs Syntactic Richness Scatter
utter_ttr = [len(set(u))/len(u) for u in corpus]
axs[1,1].scatter(utter_ttr, num_dependents, alpha=0.6, color='purple')
axs[1,1].set_title("Lexical vs Syntactic Richness")
axs[1,1].set_xlabel("Type-Token Ratio (Lexical Diversity)")
axs[1,1].set_ylabel("Number of Dependents")

plt.tight_layout()
plt.savefig("CFBL_Language_Richness_Metrics.png", dpi=300)
plt.show()

