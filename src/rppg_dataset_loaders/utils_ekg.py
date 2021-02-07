import numpy as np
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


def estimate_hr_from_ppg(ppg_signal, fps):  # in hertz
    hr_hz = -1.0  # TODO: use freq_interbeatwelch(input_signal=ppg_signal, fps=fps)
    return hr_hz
