import xml
from xml.dom import minidom

path = "dataset/CGHD-1152/C1_D1_P1.xml"
import xml.etree.ElementTree as ET
from pathlib import Path
def parsethefile(listOfFiles):
    for myFile in listOfFiles.iterdir():
        filePath = myFile
        parser = ET.XMLParser(encoding="utf-8")
        targetTree = ET.parse(filePath, parser=parser)
        rootTag = targetTree.getroot()
        width = int(rootTag.getchildren()[2][0].text)
        height = int(rootTag.getchildren()[2][1].text)
        xmin = int(rootTag.getchildren()[3][1][0].text)
        xmax = int(rootTag.getchildren()[3][1][1].text)
        ymin = int(rootTag.getchildren()[3][1][2].text)
        ymax = int(rootTag.getchildren()[3][1][3].text)
        category = 0
# Replace this with the class label of your object
parsethefile(Path(path))
# convertLabels(xmin, ymin, xmax, ymax, height, width, category)