import os
from os import walk
from detecto.core import Dataset, DataLoader, Model
from detecto.visualize import show_labeled_image
from detecto.utils import read_image
import shutil
from tqdm import tqdm
import random
import math
import pickle
import xml


base_path = "dataset/CGHD-1152"
list_of_components = set(['junction', 'text', 'gnd', 'terminal', 'crossover',
                          'resistor', 'capacitor', 'inductor',
                          'diode', 'transistor',
                          'voltage', 'current']) # only Components we want to work with
tmp = ['terminal', 'diode', 'capacitor.polarized', 'diode.light_emitting', 'voltage.dc', 'gnd', 'voltage.dc_ac', 'inductor', 'resistor', 'transistor', 'capacitor.unpolarized', 'resistor.photo', 'crossover', 'text', 'junction', 'resistor.adjustable']
clean_data_path = "dataset/clean_dataset" # create a new dataset with only "list_of_components" we want it

print("prerun running now...")
""" get list of files """
file_list = []
for (dirpath, dirnames, filenames) in walk(base_path):
    file_list.extend(filenames)
    break

""" load data to label image"""
# dataset = Dataset(base_path)
# loader = DataLoader(dataset, batch_size=2, shuffle=True)


def data_plot(file_name):
    idx = file_list.index(file_name) / 2
    image, targets = dataset[idx]
    show_labeled_image(image, targets['boxes'], targets['labels'])

""" save only data with basic component 
    will save a list of all the type, of the basic component we want """
def clean_data():
    global file_list
    dataset = Dataset(base_path)
    # remove old data
    shutil.rmtree(clean_data_path)
    os.mkdir(clean_data_path)

    wanted_components = []
    sum = 0
    print("cleaning dataset")
    for i in tqdm(range(len(dataset))):
        image, targets = dataset[i]
        labels_text = targets["labels"]
        labels_text = set(labels_text)  # remove Duplicates

        # chack if we want this image
        tosave = True
        for word in labels_text:
            if word.split('.')[0] not in list_of_components:
                tosave = False
                break
            else:
                wanted_components.append(word)

        wanted_components = list(set(wanted_components))
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

    print("save vector of wanted component to: " + clean_data_path + "/wanted_component.txt")
    with open("dataset" + "/wanted_component.txt", "wb") as fp:
        pickle.dump(wanted_components, fp)


""" create clean data with train/test/valid 
    run after create clean data only"""
TRAIN_SIZE = 0.8
TEST_SIZE = 0.1
VALID_SIZE = 0.1
SEED = 1220
def organise_data():
    clean_file_list = []
    for (dirpath, dirnames, filenames) in walk(clean_data_path):
        clean_file_list.extend(filenames)
        break

    file_name = []
    for file in clean_file_list:
        split = file.split(".")
        if split[1] == "xml":
            file_name.append(split[0])

    # shuffle and organise data to train/test/valid
    random.seed(SEED)
    random.shuffle(file_name)
    L = len(file_name)
    train_list = file_name[0:math.floor(L * TRAIN_SIZE)]
    test_list = file_name[math.floor(L * TRAIN_SIZE)+1 : math.floor(L * TRAIN_SIZE) + math.floor(L * TEST_SIZE)]
    valid_list = file_name[math.floor(L * (TRAIN_SIZE+TEST_SIZE))+1 : -1]

    os.mkdir(clean_data_path + "/train")
    os.mkdir(clean_data_path + "/test")
    os.mkdir(clean_data_path + "/valid")

    #copy files to test/train/valid
    for name in train_list:
        original = clean_data_path + "/" + name + ".jpg"
        target = clean_data_path + "/train/" + name + ".jpg"
        shutil.move(original, target)
        original = clean_data_path + "/" + name + ".xml"
        target = clean_data_path + "/train/" + name + ".xml"
        shutil.move(original, target)

    for name in test_list:
        original = clean_data_path + "/" + name + ".jpg"
        target = clean_data_path + "/test/" + name + ".jpg"
        shutil.move(original, target)
        original = clean_data_path + "/" + name + ".xml"
        target = clean_data_path + "/test/" + name + ".xml"
        shutil.move(original, target)

    for name in valid_list:
        original = clean_data_path + "/" + name + ".jpg"
        target = clean_data_path + "/valid/" + name + ".jpg"
        shutil.move(original, target)
        original = clean_data_path + "/" + name + ".xml"
        target = clean_data_path + "/valid/" + name + ".xml"
        shutil.move(original, target)

def train():
    with open("dataset" + "/wanted_component.txt", "rb") as fp:
        labels = pickle.load(fp)

    dataset_train = Dataset(clean_data_path + "/train")
    #loader = DataLoader(dataset_train, batch_size=4, shuffle=True)

    print("ready model")
    model = Model(labels)
    print("training")
    model.fit(dataset_train)

    from datetime import date

    today = date.today()

    # dd/mm/YY
    d1 = today.strftime("%d_%m_%Y")
    print("model ready at - ", d1)
    model.save('saved_models/model_alex50_' + d1 +'.pth')

    return model


if __name__ == '__main__':
    clean_data()
    # organise_data()
    # plot()

    """model = train()
    # test
    image = read_image(clean_data_path + '/test/Cd_D2_P4.jpg')
    labels, targets, scores = model.predict(image)
    show_labeled_image(image, targets['boxes'], targets['labels'])"""

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