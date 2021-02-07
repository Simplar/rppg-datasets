import csv
import os

import numpy
import pyedflib
from lxml import etree

from . import utils_ekg
from .loader_base import DatasetLoader
from .loader_base import RegularFPSChannel
from .loader_base import VideoAndPPGSession
from .loader_base import VideoChannel


class MahnobDatasetLoader(DatasetLoader):

    # abstract - implementations

    def _get_session_records(self):
        return self._prv_get_mahnob_session_records()

    def _get_session_class(self):
        return MahnobSession

    # private

    def _prv_get_mahnob_session_records(self):
        mahnob_sessions_dict = {}
        with open(os.path.join(self.path, 'metadata.csv'), 'rt', encoding='utf8') as csvfile:
            session_reader = csv.DictReader(csvfile, delimiter=',')
            for line in session_reader:
                session_record = line
                session_key = session_record['basedir']

                session_path = os.path.join(self.path, session_record['basedir'])
                if os.path.exists(session_path):
                    mahnob_sessions_dict[session_key] = session_record
        return mahnob_sessions_dict

    # public


class MahnobBDFChannel(RegularFPSChannel):

    # abstract - implementations

    def _get_raw_metadata(self):
        with pyedflib.EdfReader(self._prv_get_bdf_path()) as e:
            channel_index = e.getSignalLabels().index(self._prv_get_channel_key())
            result = {
                'main_header': e.getHeader(),
                'channel_key': self._prv_get_channel_key(),
                'channel_index': channel_index,
                'signal_header': e.getSignalHeader(channel_index),
                'sample_frequency': e.samplefrequency(channel_index),
                'frames_count': e.samples_in_file(channel_index),
                'subject': self._prv_get_subject_key()
            }
            return result

    def _get_channel_data(self):
        return self._prv_get_channel_data()

    def _get_sync_time_offset(self):
        return 0  # todo

    # private

    def _prv_get_channel_key(self):
        return self.channel_record['channel_key']

    def _prv_get_subject_key(self):  # TODO Koster: integrate 'subject' field to session_record
        video_path = self.session_metadata['video_path']
        start_idx = video_path.find('P') + 1
        end_idx = video_path.find('-')
        if end_idx <= start_idx:
            return 'unknown'
        return video_path[start_idx:end_idx]

    def _prv_get_bdf_path(self):
        return self.session_metadata['bdf_path']

    def _prv_get_channel_data(self):
        channel_data = []
        signal = numpy.zeros((self.get_frames_count(),), dtype='float64')
        channel_index = self.get_metadata()['channel_index']
        sample_frequency = self.get_metadata()['sample_frequency']
        with pyedflib.EdfReader(self._prv_get_bdf_path()) as e:
            e.readsignal(channel_index, 0, self.get_frames_count(), signal)
            for i in range(len(signal)):
                channel_data.append({'index': i, 'time': (i / sample_frequency), 'data': signal[i]})
        return channel_data

    # public

    def __init__(self, session_metadata, channel_record):
        super().__init__(session_metadata, channel_record, channel_record['channel_key'].__str__)
        self.channel_index = None


class MahnobVideoChannel(VideoChannel):

    # abstract - implementations

    def _get_sync_time_offset(self):
        return 0  # todo

    # public

    def __init__(self, session_metadata):
        super().__init__(session_metadata, None)


class MahnobSession(VideoAndPPGSession):

    # abstract - implementations

    def _get_raw_metadata(self):
        with open(os.path.join(self.get_path(), 'session.xml'), 'rt', encoding='utf8') as xml_file:
            xml_data = xml_file.read()
            root = etree.fromstring(xml_data)
            return dict(root.attrib)

    def _get_metadata(self):
        return {
            'session_record':   self.get_session_record(),
            'path':             self.get_path(),
            'bdf_path':         self._prv_get_bdf_path(),
            'video_path':       self.get_video_path(),
            'duration':         self.get_session_record()['duration'],
        }

    def _instaniate_channel(self, channel_type, channel_record):
        if channel_type == 'bdf':
            return self._prv_instaniate_bdf_channel(channel_record)
        return super()._instaniate_channel(channel_type, channel_record)

    def _get_video_channel_class(self):
        return MahnobVideoChannel

    def _get_video_path(self):
        return os.path.join(self.get_path(), self.get_session_record()['video'] + '.avi')

    def _get_signal_channel_for_vs_cross(self):
        return self._prv_get_bdf_channel({'channel_key': 'EXG1'})

    def _get_estimated_hr_by_sync_time(self, sync_time, time_duration):
        hr_channels_names = ['EXG1', 'EXG2', 'EXG3']
        hr_channels = map(
            lambda channel_name: self._prv_get_bdf_channel({'channel_key': channel_name}), hr_channels_names
        )
        hr_channels_estimates = []
        for channel in hr_channels:
            sample_frequency = channel.get_sample_frequency()
            frames_in_range = channel.get_frames_by_sync_time(sync_time, time_duration)
            frames_count_in_range = len(frames_in_range)
            signal = numpy.zeros((frames_count_in_range,), dtype='float64')
            for i in range(len(frames_in_range)):
                signal[i] = frames_in_range[i]['data']
            hr_channels_estimates.append(utils_ekg.estimate_hr_and_peaks(sample_frequency, signal))
        return utils_ekg.find_best_hr_estimation(hr_channels_estimates)

    # private

    def _prv_get_video_duration(self):
        return float(self.get_raw_metadata()['cutLenSec'])  # fast, but no so accurate - MAHNOB specific way

    def _prv_get_bdf_path(self):
        return os.path.join(self.get_path(), self.session_record['bdf'] + '.bdf')

    def _prv_instaniate_bdf_channel(self, channel_record):
        return MahnobBDFChannel(self.get_metadata(), channel_record)

    def _prv_get_bdf_channel(self, channel_record):
        return self.get_channel('bdf', channel_record)

    # public
