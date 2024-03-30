from utils import (read_video,save_video,measure,draw_player_stats,convert_pixel_distance_to_metres )
from trackers import PlayerTracker,BallTracker
from court_line_detection import CourtLineDetector
from minicourt import MiniCourt
import cv2
from copy import deepcopy
import pandas as pd
import constants

def main():
    input_video_path = "input videos\input_video.mp4"
    video_frames = read_video(input_video_path)

    player_tracker = PlayerTracker(model_path = 'yolov8x')
    ball_tracker = BallTracker(model_path="models/yolo5_last.pt")

    player_detections = player_tracker.detect_frames(video_frames,read_from_stub=True,stub_path="tracker_stubs/player_detections.pkl")
    ball_detections = ball_tracker.detect_frames(video_frames,read_from_stub=False,stub_path="tracker_stubs/ball_detections.pkl")
    ball_detections = ball_tracker.interpolate_ball_positions(ball_detections)

    court_model_path = "models/keypoints_model.pth"
    court_line_detector = CourtLineDetector(court_model_path)
    court_keypoints = court_line_detector.predict(video_frames[0])

    player_detections = player_tracker.choose_and_filter_players(court_keypoints,player_detections)

    minicourt = MiniCourt(video_frames[0])
    ball_shot_frames = ball_tracker.get_ball_shot_frames(ball_detections)

    player_minicourt_detections,ball_minicourt_detections = minicourt.convert_bboxes_to_minicourt_coordinates(player_detections,ball_detections)

    player_stats_data = [{
        'frame_num':0,
        'player_1_number_of_shots':0,
        'player_1_total_shot_speed':0,
        'player_1_last_shot_speed':0,
        'player_1_total_player_speed':0,
        'player_1_last_player_speed':0,

        'player_2_number_of_shots':0,
        'player_2_total_shot_speed':0,
        'player_2_last_shot_speed':0,
        'player_2_total_player_speed':0,
        'player_2_last_player_speed':0,
    }]

    for ball_shot_ind in range(len(ball_shot_frames)-1):
        start_frame = ball_shot_frames[ball_shot_ind]
        end_frame = ball_shot_frames[ball_shot_ind+1]
        ball_shot_time_secs = (end_frame - start_frame)/24

        distance_covered_by_ball_pixels = measure(ball_minicourt_detections[start_frame][1],ball_minicourt_detections[end_frame][1])
        distance_covered_by_ball_metres = convert_pixel_distance_to_metres(distance_covered_by_ball_pixels,constants.DOUBLE_LINE_WIDTH,minicourt.get_width_of_minicourt())

        speed_of_ball_shot = distance_covered_by_ball_metres/ball_shot_time_secs * 3.6

        player_positions = player_detections[start_frame]
        player_shot_ball = min(player_positions.keys(),key = lambda player_id: measure(player_positions[player_id],ball_minicourt_detections[start_frame][1]))


        opponent_player_id = 1 if player_shot_ball == 2 else 2
        distance_covered_by_opponent_in_pixels =  measure(player_minicourt_detections[start_frame][opponent_player_id],player_minicourt_detections[end_frame][opponent_player_id])
        distance_covered_by_opponent_in_metres = convert_pixel_distance_to_metres(distance_covered_by_opponent_in_pixels,constants.DOUBLE_LINE_WIDTH,minicourt.get_width_of_minicourt())

        speed_of_opponent = distance_covered_by_opponent_in_metres/ball_shot_time_secs * 3.6

        current_player_stats = deepcopy(player_stats_data[-1])
        current_player_stats['frame_num'] = start_frame
        current_player_stats[f"player_{player_shot_ball}_number_of_shots"] += 1
        current_player_stats[f"player_{player_shot_ball}_total_shot_speed"] += speed_of_ball_shot
        current_player_stats[f"player_{player_shot_ball}_last_shot_speed"] = speed_of_ball_shot

        current_player_stats[f"player_{opponent_player_id}_total_player_speed"] += speed_of_opponent
        current_player_stats[f"player_{opponent_player_id}_last_player_speed"] = speed_of_opponent

        player_stats_data.append(current_player_stats)

    player_stats_data_df = pd.DataFrame(player_stats_data)
    frames_df = pd.DataFrame({'frame_num':list(range(len(video_frames)))})
    player_stats_data_df = pd.merge(frames_df,player_stats_data_df,on='frame_num',how='left')
    player_stats_data_df = player_stats_data_df.ffill()

    player_stats_data_df['player_1_average_shot_speed'] = player_stats_data_df['player_1_total_shot_speed']/player_stats_data['player_1_number_of_shots']
    player_stats_data_df['player_2_average_shot_speed'] = player_stats_data_df['player_2_total_shot_speed']/player_stats_data['player_2_number_of_shots']
    player_stats_data_df['player_1_average_player_speed'] = player_stats_data_df['player_1_total_player_speed']/player_stats_data_df['player_1_number_of_shots']
    player_stats_data_df['player_2_average_player_speed'] = player_stats_data_df['player_2_total_player_speed']/player_stats_data_df['player_2_number_of_shots']
 
    output_video_frames = player_tracker.draw_bboxes(video_frames,player_detections)
    output_video_frames = ball_tracker.draw_bboxes(output_video_frames,ball_detections)
    output_video_frames = court_line_detector.draw_keypoints_on_videoframes(output_video_frames,court_keypoints)
    output_video_frames = minicourt.draw_mini_court(output_video_frames)
    output_video_frames = minicourt.draw_points_on_minicourt(output_video_frames,player_minicourt_detections)
    output_video_frames = minicourt.draw_points_on_minicourt(output_video_frames,ball_minicourt_detections,color=(0,255,255))

    output_video_frames = draw_player_stats(output_video_frames,player_stats_data_df)

    for i,frame in enumerate(output_video_frames):
        cv2.putText(frame,f"Frame {i}",(10,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

    save_video(output_video_frames,"output_videos\output_video.avi")
if __name__ == "__main__":
    main()