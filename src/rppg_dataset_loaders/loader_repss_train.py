import os

from .loader_base import DatasetLoader
from .loader_base import VideoAndPPGSession
from .loader_base import VideoChannel


class REPSS_TRAINDatasetLoader(DatasetLoader):

    # abstract - implementations

    def _get_session_records(self):
        return self._prv_get_repsstrain_session_records()

    def _get_session_class(self):
        return REPSS_TRAINSession

    # private

    def _prv_get_repsstrain_session_records(self):
        repsstrain_sessions_dict = {}
        for root, dirs, files in os.walk(self.path):
            while len(dirs) > 0:
                session_prefix_key = dirs.pop()
                if session_prefix_key.isdigit():  # Exclude foreigh directories
                    session_prefix_path = os.path.join(self.path, session_prefix_key)
                    csv_path = os.path.join(session_prefix_path, 'gt.csv')
                    if os.path.exists(session_prefix_path) and os.path.exists(csv_path):
                        with open(csv_path, 'rt', encoding='utf8') as csvfile:
                            session_info = [line.split(',') for line in csvfile]
                            session_info = list(
                                map(lambda lst_inner: list(map(lambda s: s.strip(), lst_inner)), session_info)
                            )
                            if len(session_info) == 3:
                                video_name_list = session_info[0]
                                for i in range(1, len(video_name_list)):
                                    video_name = video_name_list[i]
                                    video_file_name = video_name + '.mp4.avi'
                                    video_path = os.path.join(session_prefix_path, video_file_name)
                                    if os.path.exists(video_path):
                                        session_key = os.path.join(session_prefix_key, video_name)
                                        session_record = {'basedir': session_prefix_key,
                                                          'video': video_file_name,
                                                          'hr_mean': session_info[1][i],
                                                          'fps_mean': session_info[2][i]}
                                        repsstrain_sessions_dict[session_key] = session_record
        return repsstrain_sessions_dict

    # public


class REPSS_TRAINVideoChannel(VideoChannel):

    # abstract - implementations

    def _get_sync_time_offset(self):
        return 0  # todo

    # public

    def __init__(self, session_metadata):
        super().__init__(session_metadata, None)


class REPSS_TRAINSession(VideoAndPPGSession):

    # abstract - implementations

    def _get_raw_metadata(self):
        with open(os.path.join(self.get_path(), 'gt.csv'), 'rt', encoding='utf8') as csvfile:
            session_info = [line.split(',') for line in csvfile]
            session_info = list(map(lambda lst_inner: list(map(lambda s: s.strip(), lst_inner)), session_info))
            return session_info

    def _get_metadata(self):
        return {
            'session_record': self.get_session_record(),
            'path': self.get_path(),
            'video_path': self.get_video_path(),
        }

    def _get_video_channel_class(self):
        return REPSS_TRAINVideoChannel

    def _get_signal_channel_for_vs_cross(self):
        return self.get_video_channel()

    def _get_estimated_hr_by_sync_time(self, sync_time, time_duration):
        return self.get_session_record()['hr_mean']

    # private

    # public
