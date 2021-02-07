import os

from .loader_base import DatasetLoader
from .loader_base import IrregularFPSChannel
from .loader_base import VideoAndPPGSession
from .loader_base import VideoChannel


class REPSS_TESTDatasetLoader(DatasetLoader):

    # abstract - to override

    def _get_session_records(self):
        return self._prv_get_repsstest_session_records()

    def _get_session_class(self):
        return REPSS_TESTSession

    # private

    def _prv_get_repsstest_session_records(self):
        repsstest_sessions_dict = {}
        for root, dirs, files in os.walk(self.path):
            while len(dirs) > 0:
                session_key = dirs.pop()
                if session_key.isdigit():  # Exclude foreigh directories
                    session_path = os.path.join(self.path, session_key)
                    if os.path.exists(session_path):
                        repsstest_sessions_dict[session_key] = {"session_path": session_path,
                                                                "session_key": session_key}
            break
        return repsstest_sessions_dict

    # public


class REPSS_TESTLandmarkChannel(IrregularFPSChannel):

    # abstract - implementations

    def _get_raw_metadata(self):
        return dict({
            'frame_timestamps': self.channel_record['frame_timestamps'],
            'frames_count':     len(self.channel_record['frame_timestamps']),
        })

    def _get_metadata(self):
        return self._get_raw_metadata()

    def _get_frames(self, frame_index_start, frames_count):
        channel_data = self._prv_get_channel_data()
        return channel_data[frame_index_start: frame_index_start + frames_count]

    def _get_frame_timestamps(self):
        return self.get_metadata()['frame_timestamps']

    def _get_sync_time_offset(self):
        return 0

    # private

    def _prv_get_landmark_path(self):
        return self.session_metadata['landmark_path']

    def _prv_get_channel_data(self):
        if self.channel_data is None:
            self.channel_data = []
            frame_timestamps = self.channel_record['frame_timestamps']
            with open(self._prv_get_landmark_path()) as fp:
                i = 0
                for line in fp:
                    ints_str = line.split()
                    ints_list = list(map(lambda int_str: int(int_str), ints_str))
                    signal = []
                    for point_index in range(len(ints_list) // 2):
                        x = ints_list[2 * point_index]
                        y = ints_list[2 * point_index + 1]
                        signal.append((x, y))
                    self.channel_data.append({'index': i, 'time': frame_timestamps[i], 'data': signal})
                    i += 1
                fp.close()
            video_frames_count = len(frame_timestamps)
            signal_frames_count = len(self.channel_data)
            assert video_frames_count == signal_frames_count, \
                self.__class__.__name__ + ': video frame number doesn\'t match signal measures number'
        return self.channel_data

    # public

    def __init__(self, session_metadata, channel_record):
        super().__init__(session_metadata, channel_record, channel_record['channel_key'].__str__)
        self.channel_data = None


class REPSS_TESTVideoChannel(VideoChannel):

    # abstract - implementations

    def _get_sync_time_offset(self):
        return 0  # todo

    # public

    def __init__(self, session_metadata):
        super().__init__(session_metadata, None)


class REPSS_TESTSession(VideoAndPPGSession):

    # abstract - implementations

    def _get_metadata(self):
        return {
            'session_record':   self.get_session_record(),
            'path':             self.get_path(),
            'landmark_path':    self._prv_get_landmark_path(),
            'video_path':       self.get_video_path()
        }

    def _instaniate_channel(self, channel_type, channel_record):
        if channel_type == 'landmark':
            frame_timestamps = self.get_video_channel().get_frame_timestamps()
            gt_channel_record = dict({
                'channel_key':          'landmark',
                'landmark_path':        self._prv_get_landmark_path(),
                'frame_timestamps':     frame_timestamps,
            })
            return self._prv_instaniate_landmark_channel(gt_channel_record)
        return super()._instaniate_channel(channel_type, channel_record)

    def _get_video_channel_class(self):
        return REPSS_TESTVideoChannel

    def _get_path(self):
        return os.path.join(self.dataset_path, self.get_session_record()['session_key'])

    def _get_video_path(self):
        return os.path.join(self.get_path(), 'video.avi')

    def _get_signal_channel_for_vs_cross(self):
        return self._prv_get_landmark_channel()

    def _get_estimated_hr_by_sync_time(self, sync_time, time_duration):
        return -1  # TODO: None

    # private

    def _prv_get_landmark_path(self):
        return os.path.join(self._get_path(), 'landmark.txt')

    def _prv_instaniate_landmark_channel(self, channel_record):
        return REPSS_TESTLandmarkChannel(self.get_metadata(), channel_record)

    def _prv_get_landmark_channel(self):
        return self.get_channel('landmark')

    # public

    def get_landmark_channel(self):
        return self._prv_get_landmark_channel()
