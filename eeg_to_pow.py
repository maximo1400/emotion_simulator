import numpy as np
from scipy import signal
from typing import Optional


def calculate_band_power(
    eeg_data: list[list[float]],
    sampling_rate: float,
    bands: dict[str, tuple[float, float]],
    group_size: int = 1,
    nperseg: Optional[int] = None,
) -> list[dict[str, float]]:
    """
    Calculate band power from EEG data.

    Args:
        eeg_data: List of EEG value arrays
        sampling_rate: Sampling rate in Hz
        bands: Dict of band names to (low_freq, high_freq) tuples
               e.g., {"alpha": (8, 13), "beta": (13, 30)}
        group_size: Number of arrays to group together before calculation
        nperseg: Length of each segment for Welch's method (default: sampling_rate * 2)

    Returns:
        List of dicts with band power values for each group
    """
    if nperseg is None:
        nperseg = int(sampling_rate * 2)

    # Group the EEG arrays
    grouped_data = []
    for i in range(0, len(eeg_data), group_size):
        group = eeg_data[i : i + group_size]
        concatenated = np.concatenate(group)
        grouped_data.append(concatenated)

    results = []
    for group_idx, data in enumerate(grouped_data):
        # Calculate power spectral density using Welch's method
        freqs, psd = signal.welch(
            data,
            fs=sampling_rate,
            nperseg=min(nperseg, len(data)),
            noverlap=nperseg // 2,
        )

        band_powers = {"group": group_idx}

        # Calculate power for each band
        for band_name, (low, high) in bands.items():
            # Find frequency indices within the band
            idx_band = np.logical_and(freqs >= low, freqs <= high)

            # Integrate PSD over the band (trapezoidal integration)
            band_power = np.trapz(psd[idx_band], freqs[idx_band])
            band_powers[band_name] = band_power

        # Calculate total power for relative values
        total_power = np.trapz(psd, freqs)
        band_powers["total"] = total_power

        results.append(band_powers)

    return results


# Example usage
if __name__ == "__main__":
    # Define your frequency bands (in Hz)
    BANDS = {
        "delta": (0.5, 4),
        "theta": (4, 8),
        "alpha": (8, 13),
        "beta": (13, 30),
        "gamma": (30, 100),
    }

    # Example: Generate some fake EEG data (replace with your actual data)
    sampling_rate = 128  # Hz
    duration = 1 / 8  # seconds per array
    num_arrays = 6

    # Simulated EEG arrays (replace with your real data)
    eeg_arrays = [
        np.random.randn(sampling_rate * duration).tolist() for _ in range(num_arrays)
    ]

    # Calculate band power, grouping every 2 arrays together
    results = calculate_band_power(
        eeg_data=eeg_arrays,
        sampling_rate=sampling_rate,
        bands=BANDS,
        group_size=2,
    )

    # Print results
    for result in results:
        print(f"\nGroup {result['group']}:")
        for band in BANDS:
            power = result[band]
            relative = (power / result["total"]) * 100
            print(f"  {band:6s}: {power:.4f} ({relative:.1f}%)")
