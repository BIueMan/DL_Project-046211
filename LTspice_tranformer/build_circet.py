from component_pleaser import *

def add_wire(component_1, component_2):
    if (component_1.loc.R == 90 or component_1.loc.R == 270):
        rotation1 = 90
    else:
        rotation1 = 0

    if (component_2.loc.R == 90 or component_2.loc.R == 270):
        rotation2 = 90
    else:
        rotation2 = 0

    #if component_1.loc.X >= component_2.loc.X:
