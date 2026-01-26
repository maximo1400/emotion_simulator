import time
import pandas as pd
import pyarrow.feather as feather

input = "emotion_data/Dreamer/dreamer_bandpower_frames.feather"


class EmotionSimulator:
    input = input
    # states = ["relaxed", "happy", "sad", "angry", "fearful", "disgusted", "surprised"]
    emotion_range = (-1.0, 1.0)
    states = {
        # Low valence, low arousal
        "sad": {"va": (-1.0, 0.0), "ar": (-1.0, 0.0)},
        # Low valence, high arousal
        "angry": {"va": (-1.0, 0.0), "ar": (0.0, 1.0)},
        "fearful": {"va": (-0.5, 0.0), "ar": (0.0, 1.0)},
        # High valence, low arousal
        "relaxed": {"va": (0.0, 1.0), "ar": (-1.0, 0.0)},
        "neutral": {"va": (-0.2, 0.2), "ar": (-0.2, 0.2)},
        # High valence, high arousal
        "happy": {"va": (0.0, 1.0), "ar": (0.0, 1.0)},
        "surprised": {"va": (0.5, 1.0), "ar": (0.5, 1.0)},
    }
    sequence = [("sad", 1), ("happy", 1), ("angry", 1), ("relaxed", 1)]
    sub_id = 1
    data_frec = 8  # Hz
    data = None

    def __init__(self):
        pass

    def load_data(self):
        self.data = feather.read_feather(self.input)

        # normalize valence and arousal to self.emotion_range
        emot_min, emot_max = self.emotion_range

        val = self.data["valence"]
        val_min = val.min()
        val_max = val.max()
        self.data["valence"] = ((val - val_min) / (val_max - val_min)) * (
            emot_max - emot_min
        ) + emot_min

        ar = self.data["arousal"]
        ar_min = ar.min()
        ar_max = ar.max()
        self.data["arousal"] = ((ar - ar_min) / (ar_max - ar_min)) * (
            emot_max - emot_min
        ) + emot_min

    def add_states(self):
        self.data["state"] = "neutral"  # Default state
        for state, ranges in self.states.items():
            va_min, va_max = ranges["va"]
            ar_min, ar_max = ranges["ar"]

            condition = (
                (self.data["valence"] >= va_min)
                & (self.data["valence"] <= va_max)
                & (self.data["arousal"] >= ar_min)
                & (self.data["arousal"] <= ar_max)
            )
            self.data.loc[condition, "state"] = state
            # print(self.data[condition].shape[0], "rows assigned to state", state)

    def output_loop(self):
        emotions = []
        durations = []
        states = []
        # print(self.data.columns)
        # print(self.data.head())
        # print(self.data["state"].value_counts())

        for state, duration in self.sequence:
            emotions.append(state)
            durations.append(duration)
            states_cond = self.data["state"] == state
            states.append(self.data[states_cond])

        for duration in durations:
            t_start = time.time()
            t_cur = t_start
            while t_cur - t_start < duration:
                row = states[durations.index(duration)].sample(n=1).iloc[0]
                # Remove the last three columns (valence, arousal, state)
                row = row.iloc[:-3]
                print(row.values.tolist())
                time.sleep(1 / self.data_frec)
                t_cur = time.time()


def main():
    simulator = EmotionSimulator()
    simulator.load_data()
    simulator.add_states()
    simulator.output_loop()


if __name__ == "__main__":
    main()
