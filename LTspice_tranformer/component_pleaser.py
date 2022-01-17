LOC = "SYMBOL res"  # SYMBOL res X Y ROTATION
NAME = "SYMATTR InstName "  # SYMATTR InstName NAME
VALUE = "SYMATTR Value"  # SYMATTR Value VALUE
BIGIN_TEXT = "Version 4\nSHEET 1 880 680\n"

""" add new component_text """
def add_bigin():
    return BIGIN_TEXT

R90_R = "WINDOW 0 0 56 VBottom 2\nWINDOW 3 32 56 VTop 2\n"
def add_R(location, name, value=None):
    text = "SYMBOL res " + location.text() + "\n"
    if location.R == 90:
        text += R90_R
    text += NAME + name + "\n"
    if value is not None:
        text += VALUE + value + "\n"

    return text


R90_C = "WINDOW 0 0 32 VBottom 2\nWINDOW 3 32 32 VTop 2\n"

def add_C(location, name, value = None):
    text = "SYMBOL cap " + location.text() + "\n"
    if location.R == 90:
        text += R90_C
    text += NAME + name + "\n"
    if value is not None:
        text += VALUE + value + "\n"

    return text


R90_L = "WINDOW 0 5 56 VBottom 2\nWINDOW 3 32 56 VTop 2\n"

def add_L(location, name, value = None):
    text = "SYMBOL ind " + location.text() + "\n"
    if location.R == 90:
        text += R90_L
    text += NAME + name + "\n"
    if value is not None:
        text += VALUE + value + "\n"

    return text


R90_V = "WINDOW 0 -32 56 VBottom 2\nWINDOW 3 32 56 VTop 2\n"
R180_V = "WINDOW 0 24 96 Left 2\nWINDOW 3 24 16 Left 2\n"
R270_V = "WINDOW 0 32 56 VTop 2\nWINDOW 3 -32 56 VBottom 2\n"

def add_V(location, name, value = None):
    text = "SYMBOL voltage " + location.text() + "\n"
    if location.R == 90:
        text += R90_V
    if location.R == 180:
        text += R180_V
    if location.R == 270:
        text += R270_V
    text += NAME + name + "\n"
    if value is not None:
        text += VALUE + value + "\n"

    return text


R90_I = "WINDOW 0 -32 40 VBottom 2\nWINDOW 3 32 40 VTop 2\n"
R180_I = "WINDOW 0 24 80 Left 2\nWINDOW 3 24 0 Left 2\n"
R270_I = "WINDOW 0 32 40 VTop 2\nWINDOW 3 -32 40 VBottom 2\n"

def add_I(location, name, value = None):
    text = "SYMBOL current " + location.text() + "\n"
    if location.R == 90:
        text += R90_I
    if location.R == 180:
        text += R180_I
    if location.R == 270:
        text += R270_I
    text += NAME + name + "\n"
    if value is not None:
        text += VALUE + value + "\n"

    return text


def add_wire(location_start, location_end):
    text = "WIRE " + location_start.text() + " " + location_end.text() + "\n"
    return text

""" help with location and the component """
class Location:
    def __init__(self, x, y, rotation=None):
        self.X = x
        self.Y = y
        self.R = rotation  # rotation is 0 or 90 for R,C,L

    def text(self):
        if self.R is not None:
            return str(self.X) + " " + str(self.Y) + " " + "R" + str(self.R)
        else:
            return str(self.X) + " " + str(self.Y)

def add_com(index, loc, type):
    index = str(index)
    if type is 'R':
        text = add_R(loc, 'R'+index)
    if type is 'C':
        text = add_C(loc, 'C'+index)
    if type is 'L':
        text = add_L(loc, 'L'+index)
    if type is 'V':
        text = add_V(loc, 'V'+index)
    if type is 'I':
        text = add_I(loc, 'I'+index)
    if type is 'J':
        return ''

    return text

# a,b is up,down / left,right.  +,-/ <-
# R,C,L,I,V are for resistor, capasotor, coil, Voltage, current
# G is for ground
# j is for junction
class component:
    def __init__(self, type, value, loc):
        self.loc = loc
        self.type = type
        self.value = value

        self.con_a = 0
        self.con_b = 0
        # for R,L
        if type is "R" or type is "L":
            if loc.R == 90:
                self.con_a = Location(loc.X - 96, loc.Y + 16)
                self.con_b = Location(loc.X - 16, loc.Y + 16)
            else:
                self.con_a = Location(loc.X + 16, loc.Y + 16)
                self.con_b = Location(loc.X + 16, loc.Y + 96)

        # for C
        if type is "C":
            if loc.R == 90 or loc.R == 270:
                self.con_a = Location(loc.X - 64, loc.Y + 16)
                self.con_b = Location(loc.X - 0, loc.Y + 16)
            else:
                self.con_a = Location(loc.X + 16, loc.Y + 0)
                self.con_b = Location(loc.X + 16, loc.Y + 64)

        # for V or I
        if type is "V" or type is "I":
            if loc.R == 90:
                self.con_a = Location(loc.X - 16, loc.Y + 0)
                self.con_b = Location(loc.X - 96, loc.Y + 0)
            if loc.R == 180:
                self.con_a = Location(loc.X - 0, loc.Y - 16)
                self.con_b = Location(loc.X - 0, loc.Y - 96)
            if loc.R == 270:
                self.con_a = Location(loc.X + 16, loc.Y + 0)
                self.con_b = Location(loc.X + 96, loc.Y + 0)
            else:
                self.con_a = Location(loc.X + 0, loc.Y + 16)
                self.con_b = Location(loc.X + 0, loc.Y + 96)

        if type is "G" or type is "J":
            self.con_a = Location(loc.X, loc.Y)
            self.con_b = Location(loc.X, loc.Y)

