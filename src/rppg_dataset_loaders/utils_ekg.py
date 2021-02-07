from typing import Tuple, Dict, Any

import numpy as np
import scipy.signal as scs
from mne.preprocessing.ecg import qrs_detector


def estimate_hr_and_peaks(sampling_frequency, signal):
    peaks = qrs_detector(sampling_frequency, signal, filter_length='3.5s')
    # noinspection PyTypeChecker
    instantaneous_rates = (sampling_frequency * 60) / np.diff(peaks)
  
    # remove instantaneous rates which are lower than 30, higher than 240
    selector = (instantaneous_rates > 30) & (instantaneous_rates < 240)
    return {'hr': float(np.nan_to_num(instantaneous_rates[selector].mean())), 'peaks': peaks}


def find_best_hr_estimation(estimated_hr_and_peaks):
    """Chooses the averate heart-rate from the estimates of 3 sensors. Avoid
    rates from sensors which are far way from the other ones."""
  
    average_rates = list(map(lambda x: x['hr'], estimated_hr_and_peaks))
    agreement = 3.  # bpm
  
    non_zero = [k for k in average_rates if int(k)]
  
    if len(non_zero) == 0: return 0  # unknown!
    elif len(non_zero) == 1: return non_zero[0]
    elif len(non_zero) == 2:
        agree = abs(non_zero[0] - non_zero[1]) < agreement
        if agree: return np.mean(non_zero)
        else:  # chooses the lowest
            return sorted(non_zero)[0]
  
    # else, there are 3 values and we must do a more complex heuristic
  
    r0_agrees_with_r1 = abs(average_rates[0] - average_rates[1]) < agreement
    # r0_agrees_with_r2 = abs(average_rates[0] - average_rates[2]) < agreement  # TODO Koster: unused
    r1_agrees_with_r2 = abs(average_rates[1] - average_rates[2]) < agreement
  
    if r0_agrees_with_r1:
        if r1_agrees_with_r2:  # all 3 agree
            return np.mean(average_rates)
        else:  # exclude r2
            return np.mean(average_rates[:2])
    else:
        if r1_agrees_with_r2:  # exclude r0
            return np.mean(average_rates[1:])
        else:  # no agreement at all pick mid-way
            return sorted(average_rates)[1]

    # TODO Koster: unreachable code
    # if r1_agrees_with_r2:
    #     if r0_agrees_with_r1:  # all 3 agree
    #         return numpy.mean(average_rates)
    #     else:  # exclude r0
    #         return numpy.mean(average_rates[1:])
    # else:
    #     if r0_agrees_with_r1:  # exclude r2
    #         return numpy.mean(average_rates[:2])
    #     else:  # no agreement at all pick middle way
    #         return sorted(average_rates)[1]


def freq_welch(input_signal: np.ndarray,
               fps: float,
               freq_range: Tuple[float, float],
               **_: Dict[str, Any]
               ) -> float:
    """
    Calculates frequency in Hz basing on Welch's method to calculate PSD and spectrum analysis.
    :param input_signal: Input 1D input_signal
    :param fps: Framerate of input_signal, in Hz
    :param freq_range: (freq_min, freq_max) range to search frequency, in Hz
    :param _: dummy params for alternative functions
    :return: Estimated frequency value, in Hz
    """
    input_signal = np.asarray(input_signal)
    overlap_rate = 0.5
    if input_signal.ndim != 1:
        raise ValueError(f'input_signal is expected to be 1-dimentional, got {input_signal.ndim}')
    # Params init
    segment_length = input_signal.shape[0]
    if segment_length == 0:
        raise ValueError('input_signal should be non-empty')
    overlap_length = int(segment_length * overlap_rate)
    # PSD calculation
    freqs, psd = scs.welch(
        x=input_signal,
        fs=fps,  # in Hz
        window='hann',
        nperseg=segment_length,  # Length of each segment
        noverlap=overlap_length,  # Number of points to overlap between segments
        detrend='constant',  # Specifies how to detrend each segment
        return_onesided=True,  # If `True`, return a one - sided spectrum for real data
        scaling='density',  # Selects between computing between V ** 2 / Hz (density) and V ** 2
        average='mean'  # Method to use when averaging periodograms
    )
    # Estimate frequency
    first = np.where(freqs > freq_range[0])[0]
    last = np.where(freqs < freq_range[1])[0]
    first_index = first[0]
    last_index = last[-1]
    range_of_interest = np.asarray(range(first_index, last_index + 1, 1))
    max_idx = first_index + np.argmax(psd[range_of_interest])
    # consider neighboring peaks
    max_indices = np.asarray([max_idx - 1, max_idx, max_idx + 1])  # get neighbours of max_idx
    max_weights_aux = psd[max_indices] - min(psd[max_indices])  # consider min(neighbours) as noise level
    max_weights = max_weights_aux / sum(max_weights_aux)  # calc weights of neghbours
    freq_hz = sum([w * freqs[idx] for idx, w in zip(max_indices, max_weights)])  # weighted sum
    # clamp to [min_freq, max_freq]
    freq_hz = np.clip(freq_hz, a_min=freq_range[0], a_max=freq_range[1]).item()
    return freq_hz
