from cv_pubsubs import webcam_pub
from cv_pubsubs import window_sub

def display(cam_num=0,
            request_size = (640, 480),
            fps_limit = 24,
            window_title = 'display',
            callbacks=[]):

    def cam_handler(frame, cam_id):
        window_sub.frame_dict[str(cam_id) + "Frame"] = frame

    cam_thread = webcam_pub.frame_handler_thread(cam_num, cam_handler, fps_limit=fps_limit, high_speed=False)

    window_sub.sub_win_loop(names = [window_title],
                            input_cams=[cam_num],
                            input_vid_global_names=[str(cam_num)+'Frame'],
                            callbacks=callbacks)

    return cam_thread