import os
import re

from .loader_base import DatasetLoader
from .loader_base import VideoAndPPGSession
from .loader_base import VideoChannel


class VIPLHRDatasetLoader(DatasetLoader):

    # abstract - implementations

    def _get_session_records(self):
        return self._prv_get_viplhr_session_records()

    def _get_session_class(self):
        return VIPLHRSession

    # private

    def _prv_get_viplhr_session_records(self):
        viplhr_sessions_dict = {}
        subject_dir_regexp = r'p(\d+)'
        scenario_dir_regexp = r'v(\d+)'
        source_dir_regexp = r'source(\d+)'
        data_path = os.path.join(self.path, 'data')
        for root, dirs, files in os.walk(data_path):
            while len(dirs) > 0:
                subject_prefix_key = dirs.pop()
                if re.match(subject_dir_regexp, subject_prefix_key):  # Exclude foreigh directories
                    subject_prefix_path = os.path.join(data_path, subject_prefix_key)
                    for root2, dirs2, files2 in os.walk(subject_prefix_path):
                        while len(dirs2) > 0:
                            scenario_prefix_key = dirs2.pop()
                            if re.match(scenario_dir_regexp, scenario_prefix_key):  # Exclude foreigh directories
                                scenario_prefix_path = os.path.join(subject_prefix_path, scenario_prefix_key)
                                for root3, dirs3, files3 in os.walk(scenario_prefix_path):
                                    while len(dirs3) > 0:
                                        source_prefix_key = dirs3.pop()
                                        # Exclude foreigh directories
                                        if re.match(source_dir_regexp, source_prefix_key):
                                            session_path = os.path.join(scenario_prefix_path, source_prefix_key)
                                            video_file_name = 'video.avi'
                                            video_path = os.path.join(session_path, video_file_name)
                                            if os.path.exists(video_path):
                                                session_key = \
                                                    f'{subject_prefix_key}_{scenario_prefix_key}_{source_prefix_key}'
                                                session_record = {
                                                    'basedir':      os.path.join('data',
                                                                                 subject_prefix_key,
                                                                                 scenario_prefix_key,
                                                                                 source_prefix_key),
                                                    'video':        video_file_name,
                                                    'subject':      subject_prefix_key,
                                                    'scenario':     scenario_prefix_key,
                                                    'source':       source_prefix_key,
                                                }
                                                viplhr_sessions_dict[session_key] = session_record
        return viplhr_sessions_dict

    # public


class VIPLHRVideoChannel(VideoChannel):

    # abstract - implementations

    def _get_sync_time_offset(self):
        return 0

    # public

    def __init__(self, session_metadata):
        super().__init__(session_metadata, None)


class VIPLHRSession(VideoAndPPGSession):

    # abstract - implementations

    def _get_metadata(self):
        session_record = self.get_session_record()
        return {
            'subject':          session_record['subject'],
            'scenario':         session_record['scenario'],
            'source':           session_record['source'],
            'session_record':   session_record,
            'path':             self.get_path(),
            'video_path':       self.get_video_path(),
        }

    def _get_video_channel_class(self):
        return VIPLHRVideoChannel

    def _get_signal_channel_for_vs_cross(self):
        return self.get_video_channel()

    def _get_estimated_hr_by_sync_time(self, sync_time, time_duration):
        raise NotImplementedError  # todo

    # private

    # public
