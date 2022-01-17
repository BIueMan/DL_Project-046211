from LTspice_tranformer.component_pleaser import *
import cv2
import numpy as np
import math
from tqdm import tqdm

TO_DRAW = True

# test draw line
# TODO - MAIN: set image AND box_list AND box_label as input
image = cv2.imread("test_ground/C11_D1_P3.jpg")
box_list = [[0.6731611725531126, 0.25595961275853607, 0.14325253665447235, 0.0681848667169872, 0.9995158, 1.0, 11], [0.0996895620697423, 0.5313299153196184, 0.14102786779403687, 0.06384982678451036, 0.99940383, 1.0, 11], [0.39300291475496796, 0.2776316969018233, 0.126119002699852, 0.06309231253046739, 0.99847776, 1.0, 11], [0.5068552141126833, 0.5350701503063503, 0.10387428849935532, 0.07100672216007584, 0.99845517, 0.99998415, 1], [0.8259014691177168, 0.561588748505241, 0.19017940759658813, 0.06183779210244354, 0.9978124, 1.0, 11], [0.3572974152078754, 0.37205836765075984, 0.07970784839830901, 0.0717129691650993, 0.9971597, 0.9999865, 8], [0.6075721492892817, 0.3602217231926165, 0.06469819890825372, 0.08873481940674155, 0.9846987, 0.99991477, 1], [0.7006217084432903, 0.5849496910446569, 0.04619896098187095, 0.0912210315858063, 0.98322713, 0.999108, 8], [0.2127488790766189, 0.5441011111987265, 0.0972110852599144, 0.14776242602812617, 0.97344166, 0.9998832, 13], [0.5822320529504826, 0.5355440684055027, 0.060968093181911265, 0.070755262123911, 0.9719987, 1.0, 11], [0.6908324423589205, 0.34561034213555486, 0.02139272972157127, 0.02993349966249968, 0.86308485, 0.9999999, 7], [0.21392252649131574, 0.7571922987699509, 0.02189973075138895, 0.025067881533974094, 0.8294356, 1.0, 7], [0.702397852351791, 0.7942947404165017, 0.021913666474191767, 0.02614977014692206, 0.8207587, 0.99999976, 7], [0.5067203288015566, 0.7983980751351306, 0.023722228251005475, 0.030404263421108847, 0.730084, 0.9999956, 7], [0.50593485565562, 0.3717717426387887, 0.022606259898135538, 0.028324933428513378, 0.72388184, 1.0, 7], [0.21622652560472488, 0.38851067423820496, 0.020578279307014065, 0.02438192932229293, 0.71109116, 0.99999785, 7], [0.2201603972598126, 0.3785791059857921, 0.020515583847698412, 0.02311000071073833, 0.58447856, 0.99998975, 7]]
box_label = ['capacitor-polarized', 'capacitor-unpolarized', 'crossover', 'diode', 'diode-light_emitting', 'gnd', 'inductor', 'junction', 'resistor', 'resistor-adjustable', 'terminal', 'text', 'transistor', 'voltage-dc', 'voltage-dc_ac']

PIXEL_CUT_THRESHOLD = 3 # the number of pixel we going 2 let them live.
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
    with open('LtCircuit.asc', 'w') as f:
        f.write(text)

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


if __name__ == '__main__':
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

    COLOR_THRESHOLD = 0.82
    # get bw image
    dim = (int(image.shape[1] / 2), int(image.shape[0] / 2))
    image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    tmp = int(np.max(gray) * COLOR_THRESHOLD)
    bw_gray = (gray < tmp) * np.max(gray)

    # clean noise
    kernel = np.ones((3, 3), np.uint8)
    bw_gray = cv2.erode(bw_gray, kernel, iterations=1)
    bw_gray = cv2.dilate(bw_gray, kernel, iterations=1)
    if TO_DRAW:
        cv2.imshow('', bw_gray)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    com_list = find_conection(bw_gray, box_list)
    build_ltspice(com_list)