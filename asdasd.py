import os
curr_path = os.getcwd()
database_path = os.path.join(curr_path, "database")
print(database_path)
faces_name = []
for path in os.listdir(database_path):   
    folder_wajah = os.path.join(database_path, path) 
    print(folder_wajah)
    if os.path.isdir(folder_wajah):  
        for gambar_wajah in os.listdir(folder_wajah):
            print(gambar_wajah)