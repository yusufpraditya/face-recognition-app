import os
curr_path = os.getcwd()
database_path = os.path.join(curr_path, "database")
faces_name = []
for path in os.listdir(database_path):   
    folder_wajah = os.path.join(database_path, path) 
    if os.path.isdir(folder_wajah):  
        for gambar_wajah in os.listdir(folder_wajah):
            #print(gambar_wajah)
            pass

for dirpath, dirname, filename in os.walk(database_path):   
        for name in filename:
            print(os.path.join(dirpath, name))
       
