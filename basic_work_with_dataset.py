import os
from os import walk
import matplotlib.pyplot as plt
from detecto.utils import read_image
from detecto.core import Dataset, DataLoader
from detecto.visualize import show_labeled_image
import shutil
from tqdm import  tqdm


base_path = "dataset/CGHD-1152"
list_of_components = set(['junction', 'text', 'gnd', 'terminal', 'crossover',
                          'resistor', 'capacitor', 'inductor',
                          'diode', 'transistor',
                          'voltage', 'current']) # only Components we want to work with
clean_data_path = "dataset/clean_dataset" # create a new dataset with only "list_of_components" we want it

""" get list of files """
file_list = []
for (dirpath, dirnames, filenames) in walk(base_path):
    file_list.extend(filenames)
    break

""" load data to label image"""
dataset = Dataset(base_path)
# loader = DataLoader(dataset, batch_size=2, shuffle=True)
idx = file_list.index("C85_D1_P1.jpg")/2
image, targets = dataset[idx]
show_labeled_image(image, targets['boxes'], targets['labels'])

""" save only data with basic component """
# remove old data
shutil.rmtree(clean_data_path)
os.mkdir(clean_data_path)

# start for
sum = 0
print("cleaing dataset")
for i in tqdm(range(len(dataset))):
    image, targets = dataset[i]
    labels_text = targets["labels"]
    labels_text = set(labels_text) # remove Duplicates

    # chack if we want this image
    tosave = True
    for word in labels_text:
        if word.split('.')[0] not in list_of_components:
            tosave = False
            break

    # copy to clean data
    if tosave:
        sum += 1

        image_file_name = file_list[i * 2]
        label_file_name = file_list[i * 2 + 1]
        # copy image
        original = base_path + "/" + image_file_name
        target = clean_data_path + "/" + image_file_name
        shutil.copyfile(original, target)
        # copy label
        original = base_path + "/" + label_file_name
        target = clean_data_path + "/" + label_file_name
        shutil.copyfile(original, target)

print("clean dataset size: ", sum)





"""
## orginize data by name only (not image/xml)
file_list = []
for (dirpath, dirnames, filenames) in walk(base_path):
    file_list.extend(filenames)
    break

file_name = []
for file in file_list:
    split = file.split(".")
    if split[1] == "xml":
        file_name.append(split[0])


for file in file_name:
    img = read_image(base_path + "/" + file + "." + img_type)
    plt.imshow(img)
    plt.show()
    break"""