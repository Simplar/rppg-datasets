import os
import re

from .loader_base import DatasetLoader, VideoAndPPGSession, VideoChannel, \
    RegularFPSChannel


class SFEDU2014DatasetLoader(DatasetLoader):

    # abstract - to override

    def _get_session_records(self):
        return self._prv_get_sfedu2014_session_records()

    def _get_session_class(self):
        return SFEDU2014Session

    # private

    # noinspection DuplicatedCode
    def _prv_get_sfedu2014_session_records(self):
        sfedu2014_sessions_dict = {}
        session_dir_regexp = r'subject(\d+)'
        for root, dirs, files in os.walk(self.path):
            while len(files) > 0:
                session_key = files.pop()
                if re.match(session_dir_regexp, session_key):  # Exclude foreigh directories
                    session_path = os.path.join(self.path, session_key)
                    if os.path.exists(session_path):
                        sfedu2014_sessions_dict[session_key] = \
                            dict({"session_path": session_path, "session_key": session_key})
            break
        return sfedu2014_sessions_dict


class SFEDU2014GroundTruthChannel(RegularFPSChannel):

    # abstract - implementations

    def _get_raw_metadata(self):
        return dict({
            'ground_truth_path':    self.channel_record['ground_truth_path'],
            'frames_count':         len(self.channel_record['frame_timestamps']),
            'sample_frequency':     self.channel_record['sample_frequency'],
        })

    def _get_channel_data(self):
        return self._prv_get_channel_data()

    def _get_sync_time_offset(self):
        return 0

    # private

    def _prv_get_channel_data(self):
        if self.channel_data is None:
            filepath = self.get_metadata()['ground_truth_path']
            with open(filepath) as fp:
                datafile = fp.readlines()
            hr_bpm = None
            for line in datafile:
                if line.startswith('-'):
                    hr_bpm = -int(line.split('\t')[0])
                    break
            assert hr_bpm is not None, f'ground truth HR value not found in {filepath}'
            self.channel_data = [{'hr_bpm': hr_bpm}]
        return self.channel_data

    # public

    def __init__(self, session_metadata, channel_record):
        super().__init__(session_metadata, channel_record, channel_record['channel_key'].__str__)
        self.channel_data = None


class SFEDU2014VideoChannel(VideoChannel):

    # abstract - implementations

    def _get_sync_time_offset(self):
        return

    # public

    def __init__(self, session_metadata):
        super().__init__(session_metadata, None)


class SFEDU2014Session(VideoAndPPGSession):

    # abstract - implementations

    def _get_metadata(self):
        return {
            'session_record':     self.get_session_record(),
            'path':               self.get_path(),
            'ground_truth_path':  self._prv_get_ground_truth_path(),
            'video_path':         self.get_video_path()
        }

    # noinspection DuplicatedCode
    def _instaniate_channel(self, channel_type, channel_record):
        # Dumb metadata without real timestamps to read values only.
        # If using correct 'ground_truth' channel, we must first load video - slow
        if channel_type == 'ground_truth_sc':
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

    def _get_video_channel_class(self):
        return SFEDU2014VideoChannel

    def _get_path(self):
        return os.path.join(self.get_dataset_path(), self.get_session_key())

    def _get_video_path(self):
        return os.path.join(self.get_path(), f'{self.session_key}.mp4')

    def _get_signal_channel_for_vs_cross(self):
        return self._prv_get_ground_truth_channel()

    def _get_estimated_hr_by_sync_time(self, sync_time, time_duration):
        return self.mean_hr

    def _get_is_valid(self):
        # Calculate of average HR during when the session is firstly read
        ground_truth_sc_channel = self._prv_get_ground_truth_sc_channel()
        mean_hr_bpm = ground_truth_sc_channel['hr_bpm']
        if mean_hr_bpm is None:
            return False
        self.mean_hr = mean_hr_bpm
        return True

    def get_ppg_channel(self):
        return self._prv_get_ppg_channel()

    # private

    def _prv_get_ground_truth_path(self):
        return os.path.join(self._get_path(), f'{self.session_key}.csv')

    def _prv_instaniate_ground_truth_channel(self, channel_record):
        return SFEDU2014GroundTruthChannel(self.get_metadata(), channel_record)

    def _prv_get_ground_truth_channel(self):
        return self.get_channel('ground_truth')

    def _prv_get_ground_truth_sc_channel(self):
        return self.get_channel('ground_truth_sc')

    def _prv_get_ppg_channel(self):
        raise FileExistsError(f'There is no PPG channel data for {self.__class__}')

    # public

    def __init__(self, session_key, session_key_escaped, dataset_path, session_record):
        super().__init__(session_key, session_key_escaped, dataset_path, session_record)
        self.mean_hr = None
