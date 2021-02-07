import csv
import os
import pickle

from . import utils_ekg
from .loader_base import DatasetLoader
from .loader_base import RegularFPSChannel
from .loader_base import VideoAndPPGSession
from .loader_base import VideoChannel


class DEAPDatasetLoader(DatasetLoader):

    # abstract - implementations

    def _get_session_records(self):
        return self._prv_get_deap_session_records()

    def _get_session_class(self):
        return DEAPSession

    # private

    def _prv_get_deap_session_records(self):
        deap_sessions_records = {}
        with open(os.path.join(self.path, 'metadata_csv', 'participant_ratings.csv'), 'rt', encoding='utf8') as csvfile:
            session_reader = csv.DictReader(csvfile, delimiter=',')
            for line in session_reader:
                session_record = line
                subject_key_raw = session_record['Participant_id']
                subject_key = 's{:02d}'.format(int(subject_key_raw))
                trial_key_raw = session_record['Trial']
                trial_key = 'trial{:02d}'.format(int(trial_key_raw))
                session_path = self.path
                video_file_name = os.path.join('face_video', subject_key, subject_key + '_' + trial_key + '.avi')
                video_path = os.path.join(session_path, video_file_name)
                signal_file_name = os.path.join('data_preprocessed_python', subject_key + '.dat')
                signal_path = os.path.join(session_path, signal_file_name)
                session_record.update({
                    'basedir': '',
                    'video': video_file_name,
                    'signal': signal_file_name,
                    'subject': subject_key,
                    'trial': trial_key,
                })
                session_key = subject_key + '_' + trial_key
                if os.path.exists(session_path) and os.path.exists(video_path) and os.path.exists(signal_path):
                    deap_sessions_records[session_key] = session_record
        return deap_sessions_records

        # public


class DEAPSignalChannel(RegularFPSChannel):

    # abstract - implementations

    def _get_metadata(self):
        return {
            "sample_frequency": 128,
            "frames_count": 8064,
        }

    def _get_channel_data(self):
        return self._prv_get_channel_data()

    def _get_sync_time_offset(self):
        return 0

    # private

    def _prv_get_trial_index(self):
        trial_str = self.session_metadata['trial']
        trial_index_str = trial_str.replace('trial', '')
        return int(trial_index_str) - 1

    def _prv_get_signal_path(self):
        return self.session_metadata['signal_path']

    def _prv_get_channel_data(self):
        channel_data = []
        signal_file_name = self._prv_get_signal_path()
        trial_index = self._prv_get_trial_index()
        channel_index = self.get_channel_record()["channel_index"]
        sample_frequency = self.get_sample_frequency()
        with open(signal_file_name, "rb") as signal_file:
            signal_data = pickle.load(signal_file, encoding="bytes")  # loading python2 pickle so encoding is not utf8
            bio_data = signal_data[b'data']
            signal = bio_data[trial_index][channel_index]
            for i in range(len(signal)):
                channel_data.append({'index': i, 'time': (i / sample_frequency), 'data': signal[i]})
        assert len(channel_data) == self._get_metadata()['frames_count'], \
            self.__class__.__name__ + ': Frames count does not match'
        # from matplotlib import pyplot as plt
        # plt.plot(signal)
        # plt.show()
        return channel_data

    # public


class DEAPVideoChannel(VideoChannel):

    # abstract - implementations

    def _get_sync_time_offset(self):
        return 0

    # public

    def __init__(self, session_metadata):
        super().__init__(session_metadata, None)


class DEAPSession(VideoAndPPGSession):

    # abstract - implementations

    def _get_metadata(self):
        session_record = self.get_session_record()
        return {
            'subject': session_record['subject'],
            'trial': session_record['trial'],
            'session_record': session_record,
            'path': self.get_path(),
            'video_path': self.get_video_path(),
            'signal_path': self._prv_get_signal_path(),
        }

    def _instaniate_channel(self, channel_type, channel_record):
        if channel_type == 'signal':
            return self._prv_instaniate_signal_channel(channel_record)
        return super()._instaniate_channel(channel_type, channel_record)

    def _get_video_channel_class(self):
        return DEAPVideoChannel

    def _get_signal_channel_for_vs_cross(self):
        return self._prv_get_ppg_channel()

    def _get_estimated_hr_by_sync_time(self, sync_time, time_duration):
        ppg_channel = self._prv_get_ppg_channel()
        ppg_data = ppg_channel.get_frames_by_sync_time(sync_time, time_duration)
        ppg_signal = [sample['data'] for sample in ppg_data]
        hr = utils_ekg.estimate_hr_from_ppg(ppg_signal, ppg_channel.get_sample_frequency())
        if not hr:
            return None
        return hr * 60.0

    def get_ppg_channel(self):
        return self._prv_get_ppg_channel()

    # private

    def _prv_instaniate_signal_channel(self, channel_record):
        return DEAPSignalChannel(self.get_metadata(), channel_record, "signal")

    def _prv_get_signal_channel(self, channel_index):
        return self.get_channel("signal", {"channel_index": int(channel_index)})

    def _prv_get_ppg_channel(self):
        return self._prv_get_signal_channel(38)

    def _prv_get_signal_path(self):
        return os.path.join(self.get_path(), self.session_record['signal'])

    # public
