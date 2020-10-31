@echo off
set python_conda="C:\Anaconda\python.exe"
set face_detection="D:\Project\Python\Project_Face\FaceRecognition_V2.0\FaceRecognitionInterface.py"
set dataset_path="D:\Project\Python\Project_Face\FaceRecognition_V2.0\data"

echo =====start Face_Detection_py=============
echo =====Please wait patiently=============
%python_conda% %face_detection% %dataset_path% 
pause