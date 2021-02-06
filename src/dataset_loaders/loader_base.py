import logging
import math
import os
from abc import ABC
from enum import Enum
from fractions import Fraction
from typing import Optional, List

import numpy

from . import utils_base, utils_ffmpeg
from .ds_title import DSTitle


class TimestampAlignment(Enum):
    LEFT = 1
    RIGHT = 2


# noinspection PyShadowingBuiltins
class DatasetLoader(object):
    loader_to_title = {'MahnobDatasetLoader': DSTitle.Mahnob,
                       'UBFCDatasetLoader': DSTitle.UBFC,
                       'VIPLHRDatasetLoader': DSTitle.VIPL,
                       'DEAPDatasetLoader': DSTitle.DEAP,
                       'REPSS_TRAINDatasetLoader': DSTitle.RePSS_Train,
                       'REPSS_TESTDatasetLoader': DSTitle.RePSS_Test,
                       'DCCSFEDUDatasetLoader': DSTitle.DCC_SFEDU,
                       'SFEDU2014DatasetLoader': DSTitle.SFEDU_2014,
                       }
    title_to_loader = {title: loader for loader, title in loader_to_title.items()}

    # private

    def _prv_get_sessions(self):
        self.sessions = {}
        self.session_records = {}
        self.session_keys_escaped = {}
        session_records = self._get_session_records()
        for session_key in session_records.keys():
            session_record = session_records[session_key]
            self._prv_add_session_record(session_key, session_record)

    def _prv_add_session_record(self, session_key, session_record):
        self.sessions[session_key] = None
        self.session_records[session_key] = session_record
        session_key_escaped = utils_base.escape_filename(session_key)
        while True:
            escaped_key_exists = False
            for key in self.session_keys_escaped.keys():
                if session_key_escaped == self.session_keys_escaped[key]:
                    escaped_key_exists = True
                    break
            if not escaped_key_exists:
                break
            session_key_escaped = session_key_escaped + '_'
        self.session_keys_escaped[session_key] = session_key_escaped

    def _prv_init_with_parent_sessions_subset(self, parent_dataset_loader, session_records_subset):
        self.sessions = parent_dataset_loader.sessions
        self.session_records = session_records_subset
        self.session_keys_escaped = parent_dataset_loader.session_keys_escaped

    def _identify_dataset_title(self) -> DSTitle:
        return DatasetLoader.loader_to_title[self.__class__.__name__]

    # public

    def __init__(self, path):
        self.ds_title: DSTitle = self._identify_dataset_title()
        self.sessions = None
        self.session_records = None
        self.session_keys_escaped = None
        self.path = path

    def get_ds_title(self) -> DSTitle:
        return self.ds_title

    def get_path(self):
        return self.path

    def get_session_keys(self):
        if self.session_records is None:
            self._prv_get_sessions()
        session_keys = self.session_records.keys()
        assert len(session_keys) != 0, self.__class__.__name__ + ': no sessions loaded'
        return list(session_keys)

    def get_sessions_count(self):
        return len(self.get_session_keys())

    def get_session_by_key(self, session_key):
        if session_key not in self.get_session_keys():
            return None
        if self.sessions[session_key] is None:
            self.sessions[session_key] = self._instaniate_session(session_key)
        session = self.sessions[session_key]
        if session.get_is_valid():
            return session
        return None

    def get_session_by_key_escaped(self, session_key_escaped):
        if self.session_keys_escaped is None:
            self._prv_get_sessions()
        for session_key in self.get_session_keys():
            if self.session_keys_escaped[session_key] == session_key_escaped:
                return self.get_session_by_key(session_key)
        return None

    def get_session_by_index(self, session_index):
        if session_index < 0 or session_index >= self.get_sessions_count():
            assert self.__class__.__name__ + ': invalid session index to instaniate'
        session_key = list(self.get_session_keys())[session_index]
        return self.get_session_by_key(session_key)

    def get_subset_loader_by_filter(self, filter):
        if callable(filter):
            session_records_subset = {}
            for session_key in self.get_session_keys():
                session_record = self.session_records[session_key]
                if filter(self, session_record, session_key):
                    session_records_subset[session_key] = session_record
            if len(session_records_subset.keys()) == 0:
                return None
            subset_loader = self.__class__(self.path)
            subset_loader._prv_init_with_parent_sessions_subset(self, session_records_subset)
            return subset_loader
        assert self.__class__.__name__ + ': invalid session filter'

    def get_subset_loader_by_auto_group(self, session_record_info_key, session_record_info_value):
        # noinspection PyUnusedLocal
        def _filter(dataset_loader, session_record, session_key):
            return hasattr(session_record, "__getitem__") and \
                   session_record_info_key in session_record and \
                   session_record_info_value == session_record[session_record_info_key]

        subset_loader = self.get_subset_loader_by_filter(_filter)
        return subset_loader

    def get_subset_loader_by_session_keys(self, session_key_list):
        # noinspection PyUnusedLocal
        def _filter(dataset_loader, session_record, session_key):
            return session_key in session_key_list

        subset_loader = self.get_subset_loader_by_filter(_filter)
        assert len(session_key_list) == subset_loader.get_sessions_count(), \
            self.__class__.__name__ + ': sessions key list length does not match session subset count'
        return subset_loader

    def get_subset_loader_by_session_keys_escaped(self, session_key_escaped_list):
        # noinspection PyUnusedLocal
        def _filter(dataset_loader, session_record, session_key):
            return dataset_loader.session_keys_escaped[session_key] in session_key_escaped_list

        subset_loader = self.get_subset_loader_by_filter(_filter)
        assert len(session_key_escaped_list) == subset_loader.get_sessions_count(), \
            self.__class__.__name__ + ': sessions key escaped list length does not match session subset count'
        return subset_loader

    def get_auto_group_distinct_info_values(self, session_record_info_key):
        session_keys = self.get_session_keys()
        if session_keys is None:
            return None
        info_values_distinct = set()
        for session_key in self.session_records:
            session_record = self.session_records[session_key]
            if (hasattr(session_record, "__getitem__")) and (session_record_info_key in session_record):
                session_record_info_value = session_record[session_record_info_key]
                info_values_distinct.add(session_record_info_value)
        return list(info_values_distinct)

    def purge_resources(self):
        self._purge_resources()
        if self.sessions is None:
            return
        for key in self.sessions:
            session = self.sessions[key]
            if session is not None:
                session.purge_resources()

    # abstract - to override

    def _get_session_records(self):  # list
        raise NotImplementedError('abstract method is not overridden')

    def _get_session_class(self):
        return None

    def _instaniate_session(self, session_key):  # Session
        if session_key in self.get_session_keys():
            session_class = self._get_session_class()
            if session_class is None:
                raise NotImplementedError('abstract method is not overridden')
            session_key_escaped = self.session_keys_escaped[session_key]
            session_record = self.session_records[session_key]
            if callable(session_class):
                session = session_class(session_key, session_key_escaped, self.path, session_record)
                return session
            assert self.__class__.__name__ + ': invalid session class to instaniate'
        assert self.__class__.__name__ + ': invalid session key to instaniate'

    def _purge_resources(self):
        return


class Session(object):

    # private

    # public

    def __init__(self, session_key, session_key_escaped, dataset_path, session_record):
        self.session_key = session_key
        self.session_key_escaped = session_key_escaped
        self.dataset_path = dataset_path
        self.session_record = session_record
        self.path = None
        self.raw_metadata = None
        self.metadata = None
        self.is_valid = None
        self.channels = {}
        # to disable log from MNE
        mne_logger = logging.getLogger('mne')
        mne_logger.addFilter(lambda record: False)

    def get_path(self):
        if self.path is None:
            self.path = self._get_path()
        return self.path

    def get_dataset_path(self):
        return self.dataset_path

    def get_raw_metadata(self):
        if self.raw_metadata is None:
            self.raw_metadata = self._get_raw_metadata()
        return self.raw_metadata

    def get_metadata(self):
        if self.metadata is None:
            self.metadata = self._get_metadata()
        return self.metadata

    def get_is_valid(self):
        if self.is_valid is None:
            self.is_valid = self._get_is_valid()
        return self.is_valid

    def get_channel(self, channel_type, channel_record='_'):
        if channel_record is None:
            channel_record = '_'
        if channel_type not in self.channels:
            self.channels[channel_type] = {}
        channel_record_key = channel_record.__str__()
        if channel_record_key not in self.channels[channel_type]:
            self.channels[channel_type][channel_record_key] = self._instaniate_channel(channel_type, channel_record)
        return self.channels[channel_type][channel_record_key]

    def get_session_key(self):
        return self.session_key

    def get_session_key_escaped(self):
        return self.session_key_escaped

    def get_session_record(self):
        return self.session_record

    def get_estimated_hr_by_sync_time(self, sync_time, time_duration):
        return self._get_estimated_hr_by_sync_time(sync_time, time_duration)

    def purge_resources(self):
        self._purge_resources()
        if self.channels is None:
            return
        for channel_type_key in self.channels:
            channels_by_type = self.channels[channel_type_key]
            if channels_by_type is None:
                continue
            for channel_key in channels_by_type:
                channel = channels_by_type[channel_key]
                if channel is not None:
                    channel.purge_resources()

    # abstract - to override

    def _get_raw_metadata(self):  # dict { }
        return self.get_session_record()

    def _get_metadata(self):  # dict { }
        raise NotImplementedError('abstract method is not overridden')

    def _get_is_valid(self):
        return True

    def _get_path(self):
        return os.path.join(self.get_dataset_path(), self.get_session_record()['basedir'])

    def _instaniate_channel(self, channel_type, channel_record):  # Channel
        raise NotImplementedError('invalid channel type for ' + self.__class__.__name__ + ': ' + channel_type)

    def _get_estimated_hr_by_sync_time(self, sync_time, time_duration):
        raise NotImplementedError('abstract method is not overridden')

    def _purge_resources(self):
        return


class SingleVideoSession(Session, ABC):

    # abstract - implementations

    def _instaniate_channel(self, channel_type, channel_record):
        if channel_type == 'video':
            return self._instaniate_video_channel()
        return super()._instaniate_channel(channel_type, channel_record)

    # private

    def _prv_get_video_sync_time(self):
        # return self._get_video_channel().get_sync_time_start() # get_metadata()['start_timestamp']
        return float(self.get_video_channel().get_sync_time_by_time(0))

    def _prv_get_video_duration(self):
        # return self._get_video_time_base() * self._get_video_channel().get_metadata()['duration_ts']
        return self.get_video_channel().get_time_duration()

    def _prv_get_signal_sync_time(self):
        if self.signal_sync_time is None:
            self.signal_sync_time = self._get_signal_channel_for_vs_cross().get_sync_time_start()
        return self.signal_sync_time

    def _prv_get_signal_duration(self):
        if self.signal_duration is None:
            self.signal_duration = self._get_signal_channel_for_vs_cross().get_time_duration()
        return self.signal_duration

    def _prv_get_vs_cross_sync_time(self):
        if self.vs_cross_sync_time is None:
            self.vs_cross_sync_time = max(self._prv_get_video_sync_time(), self._prv_get_signal_sync_time())
        return self.vs_cross_sync_time

    def _prv_get_vs_cross_duration(self):
        if self.vs_cross_duration is None:
            self.vs_cross_duration = min(self._prv_get_video_sync_time() + self._prv_get_video_duration(),
                                         self._prv_get_signal_sync_time() + self._prv_get_signal_duration()) - \
                                     self._prv_get_vs_cross_sync_time()
        return self.vs_cross_duration

    def _prv_get_vs_cross_metadata(self):
        if self.vs_cross_metadata is None:
            self.vs_cross_metadata = {
                "video_sync_time": self._prv_get_video_sync_time(),
                "signal_sync_time": self._prv_get_signal_sync_time(),
                "video_duration": self._prv_get_video_duration(),
                "signal_duration": self._prv_get_signal_duration(),
                "vs_cross_sync_time": self._prv_get_vs_cross_sync_time(),
                "vs_cross_duration": self._prv_get_vs_cross_duration(),
            }
        return self.vs_cross_metadata

    # public

    def __init__(self, session_key, session_key_escaped, dataset_path, session_record):
        super().__init__(session_key, session_key_escaped, dataset_path, session_record)
        self.video_path = None
        self.signal_sync_time = None
        self.signal_duration = None
        self.vs_cross_sync_time = None
        self.vs_cross_duration = None
        self.vs_cross_metadata = None

    def get_video_path(self):
        if self.video_path is None:
            self.video_path = self._get_video_path()
        return self.video_path

    def get_video_channel(self):
        return self.get_channel('video')

    def get_video_time_base(self):
        video_channel = self.get_video_channel()
        video_metadata = video_channel.get_metadata()
        return Fraction(video_metadata['time_base'])

    def get_vs_cross_metadata(self):
        return self._prv_get_vs_cross_metadata()

    def get_vs_cross_sync_time(self):
        return self._prv_get_vs_cross_sync_time()

    def get_vs_cross_duration(self):
        return self._prv_get_vs_cross_duration()

    # abstract - to override

    def _get_video_path(self):
        session_record = self.get_session_record()
        return os.path.join(self.get_path(), session_record['video'])

    def _get_video_channel_class(self):
        raise NotImplementedError('abstract method is not overridden')

    def _instaniate_video_channel(self):
        video_channel_class = self._get_video_channel_class()
        if video_channel_class is None:
            raise NotImplementedError('abstract method is not overridden')
        if callable(video_channel_class):
            video_channel = video_channel_class(self.get_metadata())
            return video_channel
        assert self.__class__.__name__ + ': invalid video channel class to instaniate'

    def _get_signal_channel_for_vs_cross(self):
        raise NotImplementedError('abstract method is not overridden')


# TODO Koster: это может быть не только PPG-сигнал. Предлагаю переименовать все упоминания PPG в Pulse,
#  кроме реализаций класса, где действительно используется PPG-сигнал
class VideoAndPPGSession(SingleVideoSession, ABC):

    # public

    min_hr: float = None  # the minimum possible HR value

    @staticmethod
    def filter_hr_values(hr_values: List[float]) -> List[float]:
        """
        Excludes HR values that are lower than expected `min_hr` border.
        @param hr_values: input values to filter
        @return: filtered list of HR values preserving their order or original list if `min_hr` is not set.
        """
        if VideoAndPPGSession.min_hr is None:
            return hr_values
        return [hr for hr in hr_values if hr >= VideoAndPPGSession.min_hr]

    def get_ppg_channel(self):
        ppg_channel_name = self._get_ppg_channel_name()
        if ppg_channel_name is None:
            return None
        return self.get_channel(ppg_channel_name)

    # abstract - to override

    # noinspection PyMethodMayBeStatic
    def _get_ppg_channel_name(self) -> Optional[str]:
        return None


class SynchronizedFrameStream(object):

    # private

    def _prv_get_frame_index_from_time_default_fps_based(self, time, alignment=TimestampAlignment.LEFT):
        sample_frequency = self.get_metadata()['sample_frequency']
        frame_index_approx = time * sample_frequency
        # TODO Koster:
        #  добавить интерполяцию PPG-сигнала (возможно, третий TimestampAlignment, CUBIC_HERMITE_INTERP),
        #  чтобы при получении значения по таймстампу вычислялось интерполированное значение.
        #  В статье, результаты которой мы пытаемся воспроизвести,
        #  использовалась "piecewise cubic Hermite interpolation" без указания параметров и пояснений.
        #  реализация в scipy:
        #  https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.CubicHermiteSpline.html
        #  Если малыми силами разобраться не получится, возьмём другой метод интерполяции.
        if alignment == TimestampAlignment.LEFT:
            return math.floor(frame_index_approx)
        else:
            return math.ceil(frame_index_approx)

    # public

    def __init__(self, session_metadata):
        self.session_metadata = session_metadata
        self.raw_metadata = None
        self.metadata = None
        self.time_start = None
        self.time_duration = None
        self.frames_count = None
        self.sync_time_offset = None

    def get_raw_metadata(self):
        if self.raw_metadata is None:
            self.raw_metadata = self._get_raw_metadata()
        return self.raw_metadata

    def get_metadata(self):
        if self.metadata is None:
            self.metadata = self._get_metadata()
        return self.metadata

    def get_time_start(self):
        if self.time_start is None:
            self.time_start = self._get_time_from_frame_index(0)
        return self.time_start

    def get_sync_time_start(self):
        return self.get_sync_time_by_time(self.get_time_start())

    def get_time_duration(self):
        if self.time_duration is None:
            if self.get_frames_count() == 0:
                self.time_duration = 0
            else:
                self.time_duration = self._get_time_from_frame_index(
                    self.get_frames_count() - 1) - self.get_time_start()
        return self.time_duration

    def get_frames_count(self):
        if self.frames_count is None:
            self.frames_count = self._get_frames_count()
        return self.frames_count

    def get_frame_index_from_sync_time(self, sync_time, alignment=TimestampAlignment.LEFT):
        return self._get_frame_index_from_time(self.get_time_by_sync_time(sync_time), alignment)

    def get_sync_time_from_frame_index(self, frame_index):
        return self.get_sync_time_by_time(self._get_time_from_frame_index(frame_index))

    def get_sync_time_offset(self):
        if self.sync_time_offset is None:
            self.sync_time_offset = self._get_sync_time_offset()
        return self.sync_time_offset

    def get_time_by_sync_time(self, sync_time):
        return sync_time - self.get_sync_time_offset()

    def get_sync_time_by_time(self, time):
        return time + self.get_sync_time_offset()

    def get_frame_by_index(self, frame_index) -> Optional[numpy.ndarray]:
        if frame_index < 0 or frame_index >= self.get_frames_count():
            return None
        frames = self.get_frames(frame_index, 1)
        if len(frames) < 1:
            return None
        else:
            return frames[0]

    def get_frames(self, frame_index_start, frames_count):
        return self._get_frames(frame_index_start, frames_count)

    def get_frame_index_from_time(self, time, alignment=TimestampAlignment.LEFT):
        return self._get_frame_index_from_time(time, alignment)

    def get_frame_by_time(self, time, alignment=TimestampAlignment.LEFT):
        frame_index = self._get_frame_index_from_time(time, alignment)
        if frame_index is None:
            return None
        return self.get_frame_by_index(frame_index)

    def get_frame_by_sync_time(self, sync_time, alignment=TimestampAlignment.LEFT):
        time = self.get_time_by_sync_time(sync_time)
        return self.get_frame_by_time(time, alignment)

    def get_frames_by_time(self, time, time_duration):
        if time_duration < 0:
            return None
        frame_index = self._get_frame_index_from_time(time, TimestampAlignment.RIGHT)
        frame_index_2 = self._get_frame_index_from_time(time + time_duration, TimestampAlignment.LEFT)
        if frame_index is None:
            frame_index = 0
        if frame_index >= self.get_frames_count():
            return None
        if frame_index_2 is None:
            frame_index_2 = self.get_frames_count() - 1
        if frame_index_2 < 0:
            return None
        if frame_index_2 < frame_index:
            return None
        frame_count = frame_index_2 - frame_index + 1
        return self.get_frames(frame_index, frame_count)

    def get_frames_by_sync_time(self, sync_time, time_duration):
        if time_duration < 0:
            return None
        return self.get_frames_by_time(self.get_time_by_sync_time(sync_time), time_duration)

    def purge_resources(self):
        self._purge_resources()
        self.raw_metadata = None
        self.metadata = None

    # abstract - to override

    def _get_raw_metadata(self):  # dict { }
        raise NotImplementedError('abstract method is not overridden')

    def _get_metadata(self):  # dict { }
        return self.get_raw_metadata()

    def _get_frames_count(self):  # int
        raise NotImplementedError('abstract method is not overridden')

    def _get_frames(self, frame_index_start, frames_count):  # dict { index, time, data }
        raise NotImplementedError('abstract method is not overridden')

    def _get_frame_index_from_time(self, time, alignment=TimestampAlignment.LEFT):  # int
        raise NotImplementedError('abstract method is not overridden')

    def _get_time_from_frame_index(self, frame_index):  # number
        raise NotImplementedError('abstract method is not overridden')

    def _get_sync_time_offset(self):
        raise NotImplementedError('abstract method is not overridden')  # number

    def _purge_resources(self):
        return


class CachedFrameStream(SynchronizedFrameStream, ABC):

    # abstract - implementations

    def _purge_resources(self):
        super()._purge_resources()
        self.cache_data = None
        self.cache_index_start = None
        self.cache_count = None

    # private

    # noinspection PyMethodMayBeStatic, PyUnusedLocal
    def _prv_get_recommended_cache_index_start(self, frame_index_start, frames_count):
        return frame_index_start

    def _prv_load_cache_page(self, frame_index_start, frames_count):
        if self.get_cache_min_page_size() is not None:
            if self.cache_index_start is None or \
                    frame_index_start < self.cache_index_start or \
                    frame_index_start + frames_count > self.cache_index_start + self.cache_count:
                self.cache_index_start = self._prv_get_recommended_cache_index_start(frame_index_start, frames_count)
                self.cache_count = min(
                    max(self.get_cache_min_page_size(), frame_index_start + frames_count - self.cache_index_start),
                    self.get_frames_count() - self.cache_index_start)
                # print(36, "_load_cache_page update", self.cache_index_start, self.cache_count,
                # self.get_frames_count())
                self.cache_data = None  # free current cache RAM before get_frames() call to prevent double pressure
                self.cache_data = super().get_frames(self.cache_index_start, self.cache_count)
                # print(36, "_load_cache_page len", len(self.cache_data))

    def _prv_get_frames_from_cache(self, frame_index_start, frames_count):
        if self.cache_index_start is None:
            return super().get_frames(frame_index_start, frames_count)
        else:
            index_start = frame_index_start - self.cache_index_start
            # print(36, "_get_frames_from_cache", frame_index_start, index_start)
            return self.cache_data[index_start: index_start + frames_count]

    # public

    def __init__(self, session_metadata):
        super().__init__(session_metadata)
        self.cache_min_page_size = None
        self.cache_data = None
        self.cache_index_start = None
        self.cache_count = None

    def get_cache_min_page_size(self):
        if self.cache_min_page_size is None:
            self.cache_min_page_size = self._get_cache_min_page_size()
        # print(36, "get_cache_min_page_size", self.cache_min_page_size)
        return self.cache_min_page_size

    def get_frames(self, frame_index_start, frames_count):
        # print(36, "get_frames", frame_index_start, frames_count)
        self._prv_load_cache_page(frame_index_start, frames_count)
        return self._prv_get_frames_from_cache(frame_index_start, frames_count)

    # abstract - to override

    def _get_cache_min_page_size(self):  # int or None, if caching disabled
        return None


class Channel(CachedFrameStream, ABC):

    # abstract - implementations

    def _get_raw_metadata(self):
        return self.channel_record

    # private

    # public

    def __init__(self, session_metadata, channel_record, title):
        super().__init__(session_metadata)
        self.channel_record = channel_record
        self.title = title

    def get_channel_record(self):
        return self.channel_record


class IrregularFPSChannel(Channel, ABC):

    # abstract - implementations

    def _purge_resources(self):
        super()._purge_resources()
        self.frame_timestamps = None

    def _get_frames_count(self):
        return len(self.get_frame_timestamps())

    def _get_frame_index_from_time(self, time, alignment=TimestampAlignment.LEFT):
        return self._prv_get_frame_index_from_time_default_timestamps_based(time, alignment)

    def _get_time_from_frame_index(self, frame_index):
        return self.get_frame_timestamps()[frame_index]

    # private

    def _prv_get_frame_index_from_time_default_timestamps_based(self, time, alignment=TimestampAlignment.LEFT):
        frame_timestamps = self.get_frame_timestamps()
        length = len(frame_timestamps)
        if len(frame_timestamps) == 0:
            return None

        first_ts = frame_timestamps[0]
        last_ts = frame_timestamps[length - 1]

        if first_ts > time:
            if alignment == TimestampAlignment.LEFT:
                return None
            elif alignment == TimestampAlignment.RIGHT:
                return 0

        if last_ts < time:
            if alignment == TimestampAlignment.RIGHT:
                return None
            elif alignment == TimestampAlignment.LEFT:
                return length - 1

        for i in range(length - 1):
            t1 = frame_timestamps[i]
            t2 = frame_timestamps[i + 1]
            if time == t1:
                return i
            elif time == t2:
                return i + 1
            elif t1 <= time <= t2:
                if alignment == TimestampAlignment.LEFT:
                    return i
                elif alignment == TimestampAlignment.RIGHT:
                    return i + 1
        return None

    # public

    def __init__(self, session_metadata, channel_record, title):
        super().__init__(session_metadata, channel_record, title)
        self.frame_timestamps = None

    def get_frame_timestamps(self):
        if self.frame_timestamps is None:
            self.frame_timestamps = self._get_frame_timestamps()
        return self.frame_timestamps

    # abstract - to override

    def _get_frame_timestamps(self):
        raise NotImplementedError('abstract method is not overridden')


class RegularFPSChannel(Channel):

    # abstract - implementations

    def _get_frames_count(self):
        return self.get_metadata()['frames_count']

    def _get_frames(self, frame_index_start, frames_count):
        return self._get_channel_data()[frame_index_start:frame_index_start + frames_count]

    def _get_frame_index_from_time(self, time, alignment=TimestampAlignment.LEFT):
        return self._prv_get_frame_index_from_time_default_fps_based(time, alignment)

    def _get_time_from_frame_index(self, frame_index):
        sample_frequency = self.get_metadata()['sample_frequency']
        return frame_index / sample_frequency

    def _get_sync_time_offset(self):
        return 0  # todo

    def _purge_resources(self):
        super()._purge_resources()
        self.channel_data = None

    # public

    def __init__(self, session_metadata, channel_record, title):
        super().__init__(session_metadata, channel_record, title)
        self.channel_data = None

    def get_channel_data(self):
        if self.channel_data is None:
            self.channel_data = self._get_channel_data()
        return self.channel_data

    def get_sample_frequency(self):
        return self.get_metadata()['sample_frequency']

    # abstract - to override

    def _get_channel_data(self):
        raise NotImplementedError('abstract method is not overridden')


class VideoChannel(IrregularFPSChannel, ABC):

    # abstract - implementations

    def _get_cache_min_page_size(self):
        return 500

    def _get_raw_metadata(self):
        return utils_ffmpeg.get_video_metadata(self._prv_get_video_path())

    def _get_frames(self, frame_index_start, frames_count):
        video_size_with_rotation = self._prv_get_video_size_with_rotation()
        width = video_size_with_rotation['width']
        height = video_size_with_rotation['height']
        # rotation = video_size_with_rotation['rotation']
        crop_rect = self.get_crop_rect()
        if crop_rect is not None:
            width = crop_rect['w']
            height = crop_rect['h']
        video_buffer = utils_ffmpeg.get_video_frames(self._prv_get_video_path(), frame_index_start, frames_count,
                                                     crop_rect=crop_rect)
        # print(367, len(video_buffer))
        buf_ptr = 0
        video_frames = []
        for i in range(0, frames_count):
            # print (36,i)
            buf_ptr_next = buf_ptr + 3 * width * height
            linear_frame = numpy.frombuffer(video_buffer[buf_ptr:buf_ptr_next], dtype=numpy.uint8)
            linear_frame.shape = (height, width, 3)
            # if rotation == 180:
            # linear_frame = numpy.flip(linear_frame, 0)
            # elif rotation == 90:
            # linear_frame = numpy.flip(linear_frame, 1)
            frame_index = frame_index_start + i
            buf_ptr = buf_ptr_next
            video_frames.append(
                {'index': frame_index, 'time': self._get_time_from_frame_index(frame_index), 'data': linear_frame}
            )
        return video_frames

    def _get_frame_timestamps(self):
        time_base = self._prv_get_video_time_base()
        frame_timestamps = []
        video_timestamps = utils_ffmpeg.get_video_frame_timestamps(self._prv_get_video_path())
        # prev_pkt_pts = 0
        for frame_info in video_timestamps:
            if 'pkt_pts' in frame_info:
                pkt_pts = frame_info['pkt_pts']
            elif 'pkt_dts' in frame_info:
                pkt_pts = frame_info['pkt_dts']
            else:
                break  # seems to be video end. Example is DEAP/face_video/s01/s01_trial01.avi
                # or maybe pkt_pts = prev_pkt_pts + 1
            frame_timestamps.append(pkt_pts * time_base)
            # more_prev = prev_pkt_pts
            # prev_pkt_pts = pkt_pts
        return frame_timestamps

    # private

    def _prv_get_video_path(self):
        return self.session_metadata[self.session_metadata_video_path_key]

    def _prv_get_video_time_base(self):
        return Fraction(self.get_raw_metadata()['time_base'])

    def _prv_get_video_size_with_rotation(self):
        metadata = self.get_metadata()
        rotation = 0
        width = int(metadata['width'])
        height = int(metadata['height'])
        if 'tags' in metadata:
            tags = metadata['tags']
            if 'rotate' in tags:
                rotation = int(tags['rotate'])
        while rotation < 0:
            rotation += 360
        while rotation > 360:
            rotation -= 360
        if rotation != 0 and rotation != 180:
            width, height = height, width
        result = {
            'width': width,
            'height': height,
            'rotation': rotation,
        }
        crop_rect = self.get_crop_rect()
        if crop_rect is not None:
            result['crop_rect'] = crop_rect
        return result

    # public

    def __init__(self, session_metadata, channel_record, title='video'):
        super().__init__(session_metadata, channel_record, title)
        if type(self.channel_record) is dict and 'video_path' in self.channel_record:
            self.session_metadata_video_path_key = self.channel_record['video_path']
        else:
            self.session_metadata_video_path_key = 'video_path'
        if type(self.channel_record) is dict and 'crop_rect' in self.channel_record:
            self.session_metadata_crop_rect_key = self.channel_record['crop_rect']
        else:
            self.session_metadata_crop_rect_key = 'crop_rect'
        self.video_path = None
        self.crop_rect = None

    def get_video_resolution(self):
        return self._prv_get_video_size_with_rotation()
        # return {'width': self.get_metadata()['width'], 'height': self.get_metadata()['height']}

    def get_crop_rect(self):
        if self.crop_rect is None:
            self.crop_rect = self._get_crop_rect()
        return self.crop_rect

    # abstract - to override

    def _get_crop_rect(self):
        if self.session_metadata_crop_rect_key in self.session_metadata:
            crop_rect = self.session_metadata[self.session_metadata_crop_rect_key]
            return crop_rect
        return None
