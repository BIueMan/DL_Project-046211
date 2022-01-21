from component_pleaser import *
import cv2
import numpy as np
import math
from tqdm import tqdm
import pickle

TO_DRAW = False

# test draw line
# TODO - MAIN: set image AND box_list AND box_label as input
image = cv2.imread("test/test_image.jpg")
box_list = pickle.load(open("boxes.dat", "rb"))
box_label = pickle.load(open("class_names.dat", "rb"))

PIXEL_CUT_THRESHOLD = 3 # the number of pixel we going 2 let them live.
POWER_THRESHOLD = 127
def find_conection(image, box_list):
    height, width = image.shape
    # 1) create smaller boxes to remove from image
    smaller_list = []
    for i in range(len(box_list)):
        box = box_list[i]
        x1 = int((box[0] - box[2] / 2.0) * width)
        y1 = int((box[1] - box[3] / 2.0) * height)
        x2 = int((box[0] + box[2] / 2.0) * width)
        y2 = int((box[1] + box[3] / 2.0) * height)
        tmp = drawen_component(box_label[box[-1]], x1 + PIXEL_CUT_THRESHOLD, y1 + PIXEL_CUT_THRESHOLD, x2 - PIXEL_CUT_THRESHOLD, y2 - PIXEL_CUT_THRESHOLD)
        smaller_list.append(tmp)

    # 1.1) remove all component. live only PIXEL_CUT_THRESHOLD
    clean_image = image
    for box in smaller_list:
        for x in range(box.x1,box.x2 + 1):
            for y in range(box.y1, box.y2 + 1):
                clean_image[y,x] = 0

    if TO_DRAW:
        cv2.imshow('', clean_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # 2) object ditector
    # get image ready
    _, thresh = cv2.threshold(clean_image, 127, 255, cv2.THRESH_BINARY)
    contours,_ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # remove text box
    index = box_label.index("text")
    tmp_list = []
    for i in range(len(box_list)):
        if box_list[i][6] != index:
            tmp_list.append(box_list[i])
    box_list = tmp_list

    print("starting object ditector")
    draw_com_list = []
    for i in tqdm(range(len(box_list))):
        box = box_list[i]
        x1 = int((box[0] - box[2] / 2.0) * width)
        y1 = int((box[1] - box[3] / 2.0) * height)
        x2 = int((box[0] + box[2] / 2.0) * width)
        y2 = int((box[1] + box[3] / 2.0) * height)

        draw_com = drawen_component(box_label[box[-1]], x1, y1, x2, y2)
        for j in range(len(contours)):
            contour = contours[j]

            for point in contour:
                pixel = point[0]
                if pixel[0] >= x1 and pixel[0] <= x2 and pixel[1] >= y1 and pixel[1] <= y2: # there is a pixel in squere
                    #TODO: component have no direction. only 0,90.
                    # need to add 180,270 direction here
                    if pixel[0] >= x2 - PIXEL_CUT_THRESHOLD:
                        R = 0
                    elif pixel[1] <= y1 + PIXEL_CUT_THRESHOLD:
                        R = 90
                    elif pixel[0] >= x1 + PIXEL_CUT_THRESHOLD:
                        R = 270
                    elif pixel[1] >= y2 - PIXEL_CUT_THRESHOLD:
                        R = 180
                    else:
                        R = 0
                    # todo end here
                    # add to possible contours and break from this contour
                    draw_com.contours.append([j, R])
                    break
        draw_com_list.append(draw_com)

    # after we have possible connection, we can connect theme all
    for i in range(len(draw_com_list)):
        draw_com1 = draw_com_list[i]
        for j in range(i+1, len(draw_com_list)):
            draw_com2 = draw_com_list[j]
            for contour1 in draw_com1.contours:
                for contour2 in draw_com2.contours:
                    if contour1[0] == contour2[0]:
                        # set R and connection here
                        draw_com_list[i].real_connection.append([j, contour1[1]])
                        draw_com_list[j].real_connection.append([i, contour2[1]])
        # after we have all the real connection for component i
        try:
            if draw_com_list[i].real_connection[0][1] == 90 or draw_com_list[i].real_connection[0][1] == 270:
                draw_com_list[i].R = 0
            else:
                draw_com_list[i].R = 90
        except:
            print("fail to load conation to this component - ", i, " could be text")
        for j in range(len(draw_com_list[i].real_connection)):
            if draw_com_list[i].real_connection[j][1] == 270 or draw_com_list[i].real_connection[j][1] == 0:
                draw_com_list[i].con_a.append(draw_com_list[i].real_connection[j])
            else:
                draw_com_list[i].con_b.append(draw_com_list[i].real_connection[j])

    return draw_com_list


def build_ltspice(com_list):
    text = add_bigin()
    # create component to draw
    lt_com_list = []
    for com in com_list:
        x = int(com.x1 / 16) * 16
        y = int(com.y1 / 16) * 16
        loc = Location(x, y, com.R)
        value = None
        name = com.name.split("-")[0]
        if name == "capacitor":
            type = 'C'
        if name == "resistor":
            type = 'R'
        if name == "inductor":
            type = 'L'
        if name == "voltage":
            type = 'V'
        if name == "gnd":
            type = 'G'
        if name == "junction" or name == "crossover" or name == "terminal": # todo: code not build to work with crossover (lack of time)
            type = 'J'
        lt_com_list.append(component(type, value, loc))

    # draw wires
    for i in range(len(lt_com_list)): # todo: this code create duplicated wires (need to fix, lack of time)
        com = lt_com_list[i]
        if lt_com_list[i].loc.R == 0 or lt_com_list[i].loc.R == 180:
            loc1 = com.con_b
        else:
            loc1 = com.con_a
        for a in com_list[i].con_a:
            # a in the index in com_list. will use it in lt_com_list:
            if a[1] == 270 or a[1] == 0: # connect to a of other component
                loc2 = lt_com_list[a[0]].con_a
            else:
                loc2 = lt_com_list[a[0]].con_b

            # draw wire
            text += add_wire(loc1, loc2)
        # same for connection b
        if lt_com_list[i].loc.R == 0 or lt_com_list[i].loc.R == 180:
            loc1 = com.con_a
        else:
            loc1 = com.con_b
        for b in com_list[i].con_b:
            # b in the index in com_list. will use it in lt_com_list:
            if b[1] == 270 or b[1] == 0:  # connect to a of other component
                loc2 = lt_com_list[b[0]].con_a
            else:
                loc2 = lt_com_list[b[0]].con_b

            # draw wire
            text += add_wire(loc1, loc2)

    # draw component
    for i in range(len(lt_com_list)):
        loc = lt_com_list[i].loc
        type = lt_com_list[i].type
        text += add_com(i, loc, type)

    # save jt_spice file
    file_name = 'LtCircuit.asc'
    with open(file_name, 'w') as f:
        f.write(text)
    print("\nLTspice circet was created and save - ", file_name)
    return text

""" 1 ----
    ------
    -----2 """
class drawen_component:
    def __init__(self, name, x1, y1, x2, y2):
        self.name = name
        self.x1 = int(x1)
        self.y1 = int(y1)
        self.x2 = int(x2)
        self.y2 = int(y2)

        self.R = None
        self.con_a = []
        self.con_b = []
        # help vector
        self.contours = [] # appand [contours, R]
        self.real_connection = []

erode_dilate_size = 3
COLOR_THRESHOLD = 0.82
if __name__ == '__main__':
    import sys
    if len(sys.argv) == 2:
      image = cv2.imread(sys.argv[1])
    elif len(sys.argv) == 3:
      image = cv2.imread(sys.argv[1])
      COLOR_THRESHOLD = float(sys.argv[2])
    elif len(sys.argv) == 4:
      image = cv2.imread(sys.argv[1])
      COLOR_THRESHOLD = float(sys.argv[2])
      erode_dilate_size = int(sys.argv[3])

    height, width, _ = image.shape
    draw_image = image
    for i in range(len(box_list)):
        box = box_list[i]
        x1 = (box[0] - box[2] / 2.0) * width
        y1 = (box[1] - box[3] / 2.0) * height
        x2 = (box[0] + box[2] / 2.0) * width
        y2 = (box[1] + box[3] / 2.0) * height

        start_point = (int(x1), int(y1))
        end_point = (int(x2), int(y2))
        color = (255, 0, 0)
        thickness = 2
        draw_image = cv2.rectangle(draw_image, start_point, end_point, color, thickness)
    if TO_DRAW:
        cv2.imshow('', draw_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
 
    # get bw image
    dim = (int(image.shape[1] / 2), int(image.shape[0] / 2))
    image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    tmp = int(np.max(gray) * COLOR_THRESHOLD)
    bw_gray = (gray < tmp) * np.max(gray)

    # clean noise
    kernel = np.ones((erode_dilate_size, erode_dilate_size), np.uint8)
    bw_gray = cv2.erode(bw_gray, kernel, iterations=1)
    bw_gray = cv2.dilate(bw_gray, kernel, iterations=1)
    if TO_DRAW:
        cv2.imshow('', bw_gray)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    # save image to show later
    cv2.imwrite('test_image_gray.png', bw_gray)

    com_list = find_conection(bw_gray, box_list)
    build_ltspice(com_list)