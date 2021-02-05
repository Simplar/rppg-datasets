import json
import shlex
import subprocess


def get_video_metadata(path_to_input_video):
    """ Finds the metadata of the input video file """
    # -v quiet
    cmd = "ffprobe -loglevel panic -hide_banner -select_streams v -print_format json -show_streams"
    args = shlex.split(cmd)
    args.append(path_to_input_video)
    # calculate_quality the ffprobe process, decode stdout into utf-8 & convert to JSON
    ffprobeOutput = subprocess.check_output(args).decode('utf-8')
    ffprobeOutput = json.loads(ffprobeOutput)

    # for example, find height and width
    # height = ffprobeOutput['streams'][0]['height']
    # width = ffprobeOutput['streams'][0]['width']

    return ffprobeOutput['streams'][0]


def get_video_frame_timestamps(path_to_input_video):
    """ Finds the timestamps of the input video file """
    # -v quiet
    cmd = "ffprobe -loglevel panic -hide_banner -select_streams v -print_format json -show_frames"
    args = shlex.split(cmd)
    args.append(path_to_input_video)
    # calculate_quality the ffprobe process, decode stdout into utf-8 & convert to JSON
    ffprobeOutput = subprocess.check_output(args).decode('utf-8')
    ffprobeOutput = json.loads(ffprobeOutput)

    return ffprobeOutput['frames']


# noinspection PyShadowingBuiltins
def get_video_frames(path_to_input_video, frame_index, frame_count, crop_rect=None):
    # you may use -vf select for accurate frame selection
    # (something like -vf 'select=gte(n\,100)' to skip the 100 first frames)
    # eq(n\,100) - exact frame
    filter = "format=rgba,select='between(n\,{0}\,{1})'". \
        format(str(frame_index), str(frame_index + frame_count - 1))
    if crop_rect is not None:
        filter = filter + ",crop='{0}:{1}:{2}:{3}'". \
            format(crop_rect['w'], crop_rect['h'], crop_rect['x'], crop_rect['y'])

    cmd = "ffmpeg -i -loglevel panic -hide_banner -vf -vsync 0 -f image2pipe -vcodec rawvideo -pix_fmt rgb24 -"
    args = shlex.split(cmd)
    args.insert(6, filter)
    args.insert(2, path_to_input_video)
    # print(args)
    # calculate_quality the ffprobe process, decode stdout into utf-8 & convert to JSON
    ffmpegOutput = subprocess.check_output(args)

    return ffmpegOutput
