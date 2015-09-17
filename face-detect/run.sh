#!/bin/bash
nohup ./face_landmark_detection_service shape_predictor_68_face_landmarks.dat \
	../data/input/ \
	../data/output/ &