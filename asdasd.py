import os
faces = os.getcwd() + '/database/'
faces_name = []
for path in os.listdir(faces):
    faces_name.append(path)

for face in faces_name:
    image_folder = faces + face + "/"
    
    for image in os.listdir(image_folder):
        image_path = image_folder + image
        print(image_path)
        try:
            img = cv.imread(image_path)
            recognizer = cv.FaceRecognizerSF.create(args.face_recognition_model,"")
            face_feature = recognizer.feature(img)
            database[os.path.splitext(image)[0]] = face_feature
            myfile = open("data.pkl", "wb")
            pickle.dump(database, myfile)
            myfile.close()
        except:
            print('error')
print("DONE SETUP")