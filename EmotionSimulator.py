import time
import pandas as pd
import pyarrow.feather as feather

input = "emotion_data/virtual/dummy_pow.feather"


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
    sequence = [("sad", 5), ("happy", 5), ("angry", 5), ("relaxed", 5)]
    data_frec = 8  # Hz
    data = None

    def __init__(self):
        pass

    def load_data(self):
        self.data = feather.read_feather(self.input)

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

    def output_loop(self):
        emotions = []
        durations = []
        states = []

        for state, duration in self.sequence:
            emotions.append(state)
            durations.append(duration)
            states_cond = self.data["state"] == state
            states.append(self.data[states_cond])

        for duration in durations:
            t_start = time.time()
            t_cur = t_start
            while t_cur - t_start < duration:
                print(
                    states[emotions.index(state)]
                    .sample(n=1)
                    .to_dict(orient="records")[0]
                )
                time.sleep(1 / self.data_frec)
                t_cur = time.time()


def main():
    simulator = EmotionSimulator()
    simulator.load_data()
    simulator.add_states()
    simulator.output_loop()


if __name__ == "__main__":
    main()
