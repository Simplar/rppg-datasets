MoviePy does not correctly allocate the timestamp of the frame even if 
the with_times = true argument is specified.
For this you need to use ffprobe.

`pkt_pts` - time in `time_base` units