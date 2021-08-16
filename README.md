Data loaders for the most popular rPPG datasets. Time and PPG data are loaded synchronously.


Comments for the next development tasks:
- MoviePy does not correctly allocate the timestamp of the frame even if 
the with_times = true argument is specified.
For this you need to use ffprobe.
- `pkt_pts` - time in `time_base` units
