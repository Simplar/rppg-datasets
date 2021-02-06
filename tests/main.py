import pprint

from src.dataset_loaders import loader_dccsfedu, loader_ubfc, loader_repss_train, loader_repss_test, loader_mahnob, \
    loader_deap, loader_viplhr
from src.dataset_loaders.loader_base import TimestampAlignment

pp = pprint.PrettyPrinter(indent=2)


def test_video_channel(video_channel):
    print("    > VIDEO CHANNEL:", video_channel.__class__.__name__)
    print("      > Video Channel Record:", video_channel.get_channel_record())

    vres = video_channel.get_video_resolution()
    print("      > Video Resolution:", vres)
    print("      > Video Duration:", video_channel.get_time_duration())

    img = video_channel.get_frame_by_index(30)['data']
    img_shape = img.shape
    assert img_shape[1] == vres['width']
    assert img_shape[0] == vres['height']

    img = video_channel.get_frame_by_time(1)['data']
    img_shape = img.shape
    assert img_shape[1] == vres['width']
    assert img_shape[0] == vres['height']


def test_session(loader, session):
    print("  > SESSION:", session.__class__.__name__)
    sk = session.get_session_key()
    ske = session.get_session_key_escaped()
    assert sk is not None
    assert ske is not None
    print("    > Dataset Path:", session.get_dataset_path())
    print("    > Key:", sk)
    print("    > Key Escaped:", ske)
    print("    > Session Record:", session.get_session_record())

    print("    > Path:", session.get_path())
    print("    > Video Path:", session.get_video_path())

    s1s = loader.get_session_by_key_escaped(ske)
    assert session == s1s

    v = session.get_video_channel()
    assert v is not None
    test_video_channel(v)

    hr = session.get_estimated_hr_by_sync_time(1.0, 12.0)
    if hr is None:
        print("    > Estimated HR: NO DATA !!!")
    else:
        print("    > Estimated HR:", hr)

    ppg = session.get_ppg_channel()
    print("    > PPG CHANNEL:", ppg.__class__.__name__)
    if ppg is None:
        print("      > NO DATA !!!")
    else:
        ppg_data = ppg.get_frame_by_index(7)
        print("      > data:", ppg_data)


def test_loader(loader):
    n = loader.get_sessions_count()
    print("DATASET LOADER:", loader.__class__.__name__)
    print("  > Path:", loader.get_path())
    print("  > Sessions count:", n)
    for i in range(1):
        loader.purge_resources()
        session = loader.get_session_by_index(i)
        test_session(loader, session)


loader = loader_dccsfedu.DCCSFEDUDatasetLoader('DCC-SFEDU')
s1 = loader.get_session_by_index(0)
fc = s1.get_ppg_channel()
fc.get_frames_count()
fr = fc.get_frames(1, 1)
vc = s1.get_video_channel()
fr = vc.get_frames(1, 1)
rd = s1.get_raw_metadata()
md = s1.get_metadata()
test_loader(loader)

loader = loader_ubfc.UBFCDatasetLoader('UBFC_DATASET/DATASET_2')
s1 = loader.get_session_by_index(1)
sc = s1.get_ppg_channel()
cd = sc.get_channel_data()
sc1 = s1._prv_get_ground_truth_channel()
cd1 = sc1.get_channel_data()
test_loader(loader)

loader = loader_repss_train.REPSS_TRAINDatasetLoader('RePSS_Train')
l2 = loader.get_subset_loader_by_session_keys_escaped(['2_video2', '3_video1'])
test_loader(loader)
l2 = loader.get_subset_loader_by_auto_group('video', 'video3.mp4.avi')
iv = loader.get_auto_group_distinct_info_values('basedir')
iv0 = iv[0]
sl = loader.get_subset_loader_by_auto_group('basedir', iv0)
sls = sl.get_session_keys()
sk = sls[0]
ss = sl.get_session_by_key(sk)
sr = ss.get_session_record()
l2_ = l2.get_subset_loader_by_filter(
    lambda loader, session_record, session_key: (float(session_record['hr_mean']) > 80))
n2 = l2.get_sessions_count()
n2_ = l2_.get_sessions_count()
e = loader.get_session_by_key_escaped('98_video22')

loader = loader_repss_test.REPSS_TESTDatasetLoader('RePSS_Test')
test_loader(loader)
session = loader.get_session_by_index(1)
lc = session.get_landmark_channel()
fr = lc.get_frame_by_index(36)
fc = lc.get_frames_count()

loader = loader_mahnob.MahnobDatasetLoader('MahnobHCI')
session = loader.get_session_by_key("Sessions/1194")
camera = session.get_channel('video')
start_time = camera.get_sync_time_start()
total_duration_time = session.get_vs_cross_duration()
end_time = start_time + total_duration_time
start_frame = camera.get_frame_index_from_sync_time(start_time, alignment=TimestampAlignment.RIGHT)
end_frame = camera.get_frame_index_from_sync_time(end_time)
test_loader(loader)

loader = loader_deap.DEAPDatasetLoader('DEAP')
test_loader(loader)
s1 = loader.get_session_by_index(1)
sc = s1._prv_get_ppg_channel()
sd = sc.get_channel_data()
cd = s1.get_vs_cross_duration()

loader = loader_viplhr.VIPLHRDatasetLoader('VIPL-HR')
test_loader(loader)

# plt.imshow()
# plt.savefig(str(time.perf_counter_ns()) + '.png')
# plt.show()

# print(s1.get_vs_cross_duration())
# print(s1.get_estimated_hr_by_sync_time(1.0, 12.0))

# s1 = loader.get_session_by_index(0)
# v = s1.get_video_channel()
# wh = v.get_video_resolution()
# print(s1.get_session_key_escaped())
# hr = s1.get_estimated_hr_by_time(0,1)
# print(hr)
# cm = s1.get_vs_cross_duration()
# print(cm)
# pp.pprint(loader.ubfc_sessions_dict)
# s1 = loader.get_session_by_key('subject3')
# sv = s1.get_video_channel() #_instaniate_channel(s1.get_metadata(), 'ground_truth', None)
# sc = s1.get_estimated_hr_by_sync_time(0,1)
# print(s1, sc)
# print(s1.get_estimated_hr_by_sync_time(1.0, 2.0))
# print(36,len(sc._get_channel_data()))


# for sk in loader.get_session_keys():
#  s = loader.get_session_by_key(sk)
#  v = s.get_video_channel()
#  fps = Fraction(v.get_metadata()['avg_frame_rate'])
#  print(sk, "fps:", float(fps), fps)
#  print(sk, v._get_frame_timestamps())
#  print(" ")
#  print(" ")

# pp.pprint(v.get_metadata())


# pp.pprint(s1._get_metadata())
# v = s1.get_video_channel()
# pp.pprint(v)
# pp.pprint(v.get_video_resolution())

# loader = loader_mahnob.MahnobDatasetLoader('MahnobHCI')
# n = loader.get_sessions_count()

# print(loader.get_session_keys())
# s1 = loader.get_session_by_key("Sessions/1956")

# print (s1.get_raw_metadata())

# print (s1.get_vs_cross_sync_time())
# print (s1.get_vs_cross_duration())
# print (s1._get_video_channel().sync_time_from_frame_index(10))

# pp.pprint(s1)

# cb = s1.get_channel('bdf', 'EXG1')
# pp.pprint(cb.get_raw_metadata())
# print(cb.get_estimate_hr_and_peaks_by_time(1.0, 20.0))
# print(s1.get_estimated_hr_by_sync_time(0.0, 4.0))
# print(s1.get_estimated_hr_by_sync_time(4.0, 4.0))
# print(s1.get_estimated_hr_by_sync_time(8.0, 4.0))
# print(s1.get_estimated_hr_by_sync_time(12.0, 4.0))
# print(s1.get_estimated_hr_by_sync_time(16.0, 4.0))
# print(s1.get_estimated_hr_by_sync_time(70.0, 20.0))
# print(s1.get_estimated_hr_by_sync_time(80.0, 20.0))
# print(s1.get_estimated_hr_by_sync_time(90.0, 20.0))
# print(s1.get_vs_cross_metadata())

# cv = s1.get_channel('video')
# fi = cv.frame_index_from_sync_time(0)
# print (fi)

# f0 = cv.get_frame_by_index(50)['data']
# ppl.imsave("1.png", f0)
# f0 = cv.get_frame_by_index(450)['data']
# ppl.imsave("11.png", f0)

# f0 = cv.get_frame_by_index(8450)['data']
# ppl.imsave("2.png", f0)

# print(cv.get_video_resolution())
# print(cv._get_frame_timestamps())
# print(cv.get_raw_metadata())
# print("2")
# img2 = cv.get_frames(0,100)[21]
# print("1")
# img1 = cv.get_frames(9,50)[12]
# print("3")
# img3 = cv.get_frames(10,12)[11]
# print("4")
# img4 = cv.get_frames(21,20)[0]
# print(img['time'])
# print(img['data'])
# plt.imshow(img['data'])
# ppl.imsave("1.png", img1['data'])
# ppl.imsave("11.png", img2['data'])
# ppl.imsave("111.png", img3['data'])
# ppl.imsave("1111.png", img4['data'])

# print(plt.imread('test.png'))
# plt.imshow(plt.imread('test.png'))
# plt.show()

# m = s1.get_video_frames(10,1)
# print len(m)

# m = s1.get_session_metadata()
# pp.pprint(m)

# m = s1._get_signal_duration()
# pp.pprint(m)

# m = s1._get_video_duration()
# pp.pprint(m)

# m = s1._get_vs_cross_start()
# pp.pprint(m)

# m = s1.get_vs_metadata()
# pp.pprint(m)


# m = s1.estimate_heartrate_in_bpm()
# pp.pprint(m)

# m = s1.get_vs_metadata()
# pp.pprint(m)

# b = loader_mahnob.MahnobBDFChannel("egk", "MahnobHCI\\Sessions\\36\\Part_1_S_Trial18_emotion.bdf", "EXG3")
# pp.pprint(b._get_frames(2,20))

# v = MahnobLoader.MahnobVideoChannel("vidos", "MahnobHCI\\Sessions\\36\\P1-Rec1-2009.07.09.17.53.46_C1 trigger _C_Section_36.avi")
# pp.pprint(v._get_frame_timestamps())
