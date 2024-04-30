import cv2
import xml.etree.ElementTree as ET
import numpy as np




def load_mask(file_path, file_name, imsize):
    try:
        mask = cv2.imdecode(np.fromfile(fr'{file_path}/{file_name}.png', np.uint8), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError()
    except:
        tree = ET.parse(fr'{file_path}/{file_name}.xml')
        root = tree.getroot()
        mask = np.zeros((imsize[0], imsize[1]), dtype=np.uint8)
        color = 1
        for region in root.findall('.//Region'):
            vertices = region.findall('.//Vertex')
            points = [(float(vertex.attrib['X']), float(vertex.attrib['Y'])) for vertex in vertices]

            # make ploy
            pts = np.array(points, np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.fillPoly(mask, [pts], color)
            color += 1
        cv2.imwrite(fr'{file_path}/{file_name}.png', mask)

    return mask




if __name__ == '__main__':
    mask = load_mask('/home/seegene/Desktop/data/MoNuSeg2018/train/Annotations', 'TCGA-18-5592-01Z-00-DX1', (1000,1000,3))