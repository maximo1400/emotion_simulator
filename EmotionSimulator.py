import time
import pyarrow.feather as feather
import queue


# fmt: off
input = "emotion_data/Dreamer/dreamer_bandpower_frames.feather"
pow_columns= [
        "AF3/theta", "AF3/alpha", "AF3/betaL", "AF3/betaH", "AF3/gamma",
        "F7/theta",  "F7/alpha",  "F7/betaL",  "F7/betaH",  "F7/gamma",
        "F3/theta",  "F3/alpha",  "F3/betaL",  "F3/betaH",  "F3/gamma",
        "FC5/theta", "FC5/alpha", "FC5/betaL", "FC5/betaH", "FC5/gamma",
        "T7/theta",  "T7/alpha",  "T7/betaL",  "T7/betaH",  "T7/gamma",
        "P7/theta",  "P7/alpha",  "P7/betaL",  "P7/betaH",  "P7/gamma",
        "O1/theta",  "O1/alpha",  "O1/betaL",  "O1/betaH",  "O1/gamma",
        "O2/theta",  "O2/alpha",  "O2/betaL",  "O2/betaH",  "O2/gamma",
        "P8/theta",  "P8/alpha",  "P8/betaL",  "P8/betaH",  "P8/gamma",
        "T8/theta",  "T8/alpha",  "T8/betaL",  "T8/betaH",  "T8/gamma",
        "FC6/theta", "FC6/alpha", "FC6/betaL", "FC6/betaH", "FC6/gamma",
        "F4/theta",  "F4/alpha",  "F4/betaL",  "F4/betaH",  "F4/gamma",
        "F8/theta",  "F8/alpha",  "F8/betaL",  "F8/betaH",  "F8/gamma",
        "AF4/theta", "AF4/alpha", "AF4/betaL", "AF4/betaH", "AF4/gamma"]

# fmt: on


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
    sequence = [("sad", 1), ("happy", 1.5), ("angry", 0.5), ("relaxed", 1)]
    sub_id = 1
    data_frec = 8  # Hz
    data = None
    out_queue = None
    emotiv_columns = pow_columns

    def __init__(self, queue: queue.Queue):
        self.out_queue = queue
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

    def update_queue(self, pow) -> None:
        # while not self.out_queue.empty():
        #     time.sleep(0.1)

        self.out_queue.put(pow)
        print(f"new data put in queue, mean: {sum(pow) / len(pow):.4f}")

        # while not self.out_queue.empty():
        #     time.sleep(0.1)

    def output_loop(self):
        pow_by_state = {}
        row_read = {}
        # print(self.data.columns)
        # print(self.data.head())
        # print(self.data["state"].value_counts())

        for state, duration in self.sequence:
            if pow_by_state.get(state) is None:
                # filter data for the given state and subject id
                s_state = self.data["state"] == state
                s_sub = self.data["subject_id"] == self.sub_id
                mask = s_state & s_sub
                data = self.data[mask].reindex(
                    columns=self.emotiv_columns
                )  # reset index
                # pow_by_state.append(data)
                pow_by_state[state] = data
                row_read[state] = 0

        for idx in range(len(self.sequence)):
            state, duration = self.sequence[idx]
            t_start = time.time()
            t_cur = t_start
            while t_cur - t_start < duration:
                # row = states[idx].sample(n=1).iloc[0] # Get a random row
                row = pow_by_state[state].iloc[row_read[state]]  # Get row in sequence

                # print(row.values.tolist())
                self.update_queue(row.values.tolist())
                time.sleep(1 / self.data_frec)
                row_read[state] += 1
                if row_read[state] >= pow_by_state[state].shape[0]:
                    row_read[state] = 0
                t_cur = time.time()


def main():
    out = queue.Queue()
    simulator = EmotionSimulator(out)
    simulator.load_data()
    simulator.add_states()
    simulator.output_loop()


if __name__ == "__main__":
    main()
