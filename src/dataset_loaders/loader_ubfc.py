import os
import re

import numpy

from .loader_base import DatasetLoader
from .loader_base import RegularFPSChannel
from .loader_base import VideoAndPPGSession
from .loader_base import VideoChannel


class UBFCDatasetLoader(DatasetLoader):

    # abstract - to override

    def _get_session_records(self):
        return self._prv_get_ubfc_session_records()

    def _get_session_class(self):
        return UBFCSession

    # private

    # noinspection DuplicatedCode
    def _prv_get_ubfc_session_records(self):
        ubfc_sessions_dict = {}
        session_dir_regexp = r'subject(\d+)'
        for root, dirs, files in os.walk(self.path):
            while len(dirs) > 0:
                session_key = dirs.pop()
                if re.match(session_dir_regexp, session_key):  # Exclude foreigh directories
                    session_path = os.path.join(self.path, session_key)
                    if os.path.exists(session_path):
                        ubfc_sessions_dict[session_key] = {"session_path": session_path, "session_key": session_key}
            break
        return ubfc_sessions_dict

    # public


class UBFCGroundTruthChannel(RegularFPSChannel):

    # abstract - implementations

    def _get_raw_metadata(self):
        return {
            'ground_truth_path':    self.channel_record['ground_truth_path'],
            'frames_count':         len(self.channel_record['frame_timestamps']),
            'sample_frequency':     self.channel_record['sample_frequency'],
        }

    def _get_channel_data(self):
        return self._prv_get_channel_data()

    def _get_sync_time_offset(self):
        return 0

    # private

    def _prv_get_channel_data(self):
        if self.channel_data is None:
            sample_frequency = self.get_metadata()['sample_frequency']
            self.channel_data = []
            ground_truth_index = self.get_channel_record()['ground_truth_index']
            with open(self.get_metadata()['ground_truth_path']) as fp:
                for i in range(ground_truth_index): __ = fp.readline()
                line = fp.readline()
                floats_str = line.split()
                signal = list(map(lambda float_str: float(float_str), floats_str))
                for i in range(len(signal)):
                    self.channel_data.append({'index': i, 'time': (i / sample_frequency), 'data': signal[i]})
                fp.close()
            video_frames_count = len(self.channel_record['frame_timestamps'])
            signal_frames_count = len(self.channel_data)
            if video_frames_count > 0:
                assert video_frames_count == signal_frames_count, \
                    self.__class__.__name__ + ': video frame number doesn\'t match signal measures number'
        return self.channel_data

    # public

    def __init__(self, session_metadata, channel_record):
        super().__init__(session_metadata, channel_record, channel_record['channel_key'].__str__)
        self.channel_data = None


class UBFCVideoChannel(VideoChannel):

    # abstract - implementations

    def _get_sync_time_offset(self):
        return 0  # todo

    # public

    def __init__(self, session_metadata):
        super().__init__(session_metadata, None)


class UBFCSession(VideoAndPPGSession):

    # abstract - implementations

    def _get_metadata(self):
        return {
            'session_record':       self.get_session_record(),
            'path':                 self.get_path(),
            'ground_truth_path':    self._prv_get_ground_truth_path(),
            'video_path':           self.get_video_path()
        }

    # noinspection DuplicatedCode
    def _instaniate_channel(self, channel_type, channel_record):
        if channel_type == 'ground_truth_sc':
            # Dumb metadata without real timestamps to read values only.
            # If using correct `ground_truth` channel, we must first load video - slow
            gt_sc_channel_record = dict({
                'channel_key':          'ground_truth',
                'ground_truth_path':    self._prv_get_ground_truth_path(),
                'ground_truth_index':   1,
                'frame_timestamps':     [],
                'sample_frequency':     1,
            })
            return self._prv_instaniate_ground_truth_channel(gt_sc_channel_record)
        if channel_type == 'ground_truth':
            frame_timestamps = self.get_video_channel().get_frame_timestamps()
            gt_channel_record = dict({
                'channel_key':          'ground_truth',
                'ground_truth_path':    self._prv_get_ground_truth_path(),
                'ground_truth_index':   1,
                'frame_timestamps':     frame_timestamps,
                'sample_frequency':     (1 / (frame_timestamps[1] - frame_timestamps[0])),
            })
            return self._prv_instaniate_ground_truth_channel(gt_channel_record)
        if channel_type == 'ground_truth_ppg':
            frame_timestamps = self.get_video_channel().get_frame_timestamps()
            ppg_channel_record = {
                'channel_key':          'ground_truth',
                'ground_truth_path':    self._prv_get_ground_truth_path(),
                'ground_truth_index':   0,
                'frame_timestamps':     frame_timestamps,
                'sample_frequency':     (1 / (frame_timestamps[1] - frame_timestamps[0])),
            }
            return self._prv_instaniate_ground_truth_channel(ppg_channel_record)
        return super()._instaniate_channel(channel_type, channel_record)

    def _get_video_channel_class(self):
        return UBFCVideoChannel

    def _get_path(self):
        return os.path.join(self.get_dataset_path(), self.get_session_key())

    def _get_video_path(self):
        return os.path.join(self.get_path(), 'vid.avi')

    def _get_signal_channel_for_vs_cross(self):
        return self._prv_get_ground_truth_channel()

    def _get_estimated_hr_by_sync_time(self, sync_time, time_duration):
        ground_truth_channel = self._prv_get_ground_truth_channel()
        ground_truth_frames = ground_truth_channel.get_frames_by_sync_time(sync_time, time_duration)
        hr_values = list(map(lambda ground_truth_frame: ground_truth_frame['data'], ground_truth_frames))
        hr_values = self.filter_hr_values(hr_values=hr_values)
        if len(hr_values) > 0:
            # return average of found HR values as `ground_truth` HR
            return numpy.mean(hr_values)
        else:
            # No appropriate HR values were found; return session average HR value
            return self.mean_hr

    def _get_is_valid(self):
        # Calculation of session average HR value when session is firstly read
        ground_truth_sc_channel = self._prv_get_ground_truth_sc_channel()
        ground_truth_frames_all = ground_truth_sc_channel.get_channel_data()
        hr_values = list(map(lambda ground_truth_frame: ground_truth_frame['data'], ground_truth_frames_all))
        hr_values = self.filter_hr_values(hr_values=hr_values)
        if len(hr_values) > 0:
            self.mean_hr = numpy.mean(hr_values)
            return True
        else:
            # All `ground_truth` values were excluded; session is considered invalid
            return False

    def get_ppg_channel(self):
        return self._prv_get_ppg_channel()

    # private

    def _prv_get_ground_truth_path(self):
        return os.path.join(self._get_path(), 'ground_truth.txt')

    def _prv_instaniate_ground_truth_channel(self, channel_record):
        return UBFCGroundTruthChannel(self.get_metadata(), channel_record)

    def _prv_get_ground_truth_channel(self):
        return self.get_channel('ground_truth')

    def _prv_get_ground_truth_sc_channel(self):
        return self.get_channel('ground_truth_sc')

    def _prv_get_ppg_channel(self):
        return self.get_channel('ground_truth_ppg')

    # public

    def __init__(self, session_key, session_key_escaped, dataset_path, session_record):
        super().__init__(session_key, session_key_escaped, dataset_path, session_record)
        self.mean_hr = None
