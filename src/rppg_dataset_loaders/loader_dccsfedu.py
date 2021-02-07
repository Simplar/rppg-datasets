import json
import os
import re

import numpy as np

from .loader_base import DatasetLoader
from .loader_base import VideoAndPPGSession
from .loader_base import VideoChannel


# TODO Koster: fix frame rotation (frames are rotated, but 'rotation' field in get_video_resolution is '0')
class DCCSFEDUDatasetLoader(DatasetLoader):
    # abstract - implementations

    def _get_session_records(self):
        return self._prv_get_dccsfedu_session_records()

    def _get_session_class(self):
        return DCCSFEDUSession

    # private

    def _prv_get_dccsfedu_session_records(self):
        json_regexp = r'(\S+)\_(\d+)\_METADATA.json'
        dccsfedu_sessions_records = {}
        for root, dirs, files in os.walk(self.path):
            while len(files) > 0:
                file_name = files.pop()
                match = re.match(json_regexp, file_name)
                if match is not None:
                    subject_key = str(match.group(1))
                    start_timestamp_key = int(str(match.group(2)))
                    json_file_name = file_name
                    video_file_name = '{0}_{1}_FACE.mp4'.format(subject_key, start_timestamp_key)
                    finger_file_name = '{0}_{1}_FINGER.mp4'.format(subject_key, start_timestamp_key)
                    is_video_file_exists = os.path.exists(os.path.join(self.path, video_file_name))
                    is_finger_file_exists = os.path.exists(os.path.join(self.path, finger_file_name))
                    if is_video_file_exists and is_finger_file_exists:
                        session_record = {
                            'basedir': '',
                            'json': json_file_name,
                            'video': video_file_name,
                            'finger': finger_file_name,
                            'subject': subject_key,
                            'start_timestamp': start_timestamp_key,
                        }
                        session_key = '{0}_{1}'.format(subject_key, start_timestamp_key)
                        dccsfedu_sessions_records[session_key] = session_record
        return dccsfedu_sessions_records


# public


class DCCSFEDUFingerChannel(VideoChannel):

    # abstract - implementations

    def _get_sync_time_offset(self):
        return 0

    # public

    def __init__(self, session_metadata):
        super().__init__(session_metadata, {"video_path": "finger_path"})


class DCCSFEDUVideoChannel(VideoChannel):

    # abstract - implementations

    def _get_sync_time_offset(self):
        return 0

    # public

    def __init__(self, session_metadata):
        super().__init__(session_metadata, None)


class DCCSFEDUSession(VideoAndPPGSession):

    # abstract - implementations

    def _get_raw_metadata(self):
        session_record = self.get_session_record()
        with open(os.path.join(self.get_path(), session_record['json']), 'rt', encoding='utf8') as json_file:
            json_data = json_file.read()
            return dict(json.loads(json_data))

    def _get_metadata(self):
        session_record = self.get_session_record()
        raw_metadata = self.get_raw_metadata()
        return {
            'subject': session_record['subject'],
            'start_timestamp': raw_metadata['start_timestamp'],
            'stop_timestamp': raw_metadata['stop_timestamp'],
            'session_record': session_record,
            'path': self.get_path(),
            'video_path': self.get_video_path(),
            'finger_path': self._prv_get_finger_path(),
        }

    def _instaniate_channel(self, channel_type, channel_record):
        if channel_type == 'finger':
            return self._prv_instaniate_finger_channel(channel_record)
        return super()._instaniate_channel(channel_type, channel_record)

    def _get_video_channel_class(self):
        return DCCSFEDUVideoChannel

    def _get_signal_channel_for_vs_cross(self):
        return self._prv_get_ppg_channel()

    def _get_estimated_hr_by_sync_time(self, sync_time, time_duration):
        # get PPG signal
        ppg_channel = self._prv_get_ppg_channel()
        ppg_data = ppg_channel.get_frames_by_sync_time(sync_time, time_duration)
        ppg_signal = np.asarray([np.mean(frame['data'], axis=(0, 1)) for frame in ppg_data])
        ppg_signal = ppg_signal[:, 1]  # use `green` component
        eps = 1e-5
        ppg_signal_norm = (ppg_signal - ppg_signal.mean(axis=0) + eps) / (ppg_signal.std(axis=0) + eps)
        ppg_signal_norm_neg = -ppg_signal_norm

        # get FPS
        # TODO: calculate FPS once for the whole signal
        timestamps = [float(frame['time']) - float(ppg_data[0]['time']) for frame in ppg_data]
        time_diff = np.diff(timestamps)
        spf = float(np.mean(time_diff))
        fps = 1.0 / spf

        freq_range = [self.min_hr_bpm / 60.0, self.max_hr_bpm / 60.0]
        prominence = 1.0
        hr_hz = VideoAndPPGSession.estimate_hr_by_ppg_signal(input_signal=ppg_signal_norm_neg,
                                                             fps=fps,
                                                             freq_range=freq_range,
                                                             prominence=prominence,
                                                             consider_neighboring_peaks=True)
        if hr_hz is None:
            return None
        return hr_hz * 60.0

    def get_ppg_channel(self):
        return self._prv_get_ppg_channel()

    # private

    # noinspection PyUnusedLocal
    def _prv_instaniate_finger_channel(self, channel_record):
        return DCCSFEDUFingerChannel(self.get_metadata())

    def _prv_get_ppg_channel(self):
        return self.get_channel("finger")

    def _prv_get_finger_path(self):
        return os.path.join(self.get_path(), self.session_record['finger'])

    # public
