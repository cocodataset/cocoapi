from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
import json
import cv2 as cv


# Import annotations (check)
# Create loop to loop through images (check)
# Show image (check)
# Allow going backwards (check)
# Modify label category
# Save modified annotations
# Save processed images
# Save weird images
# Print image progress after showing 10 images

def box_cxcywh_to_xyxy(x):
    # Converts bounding boxes to (x1, y1, x2, y2) coordinates of top left and bottom right corners
    x_c, y_c, w, h = x
    b = [(x_c), (y_c),
        (x_c + w), (y_c + h)]
    box = list(map(int, map(round, b)))
    return box

def save_dataset(imgs, anns, filename):

    # Load dataset val to get structure
    test = '../annotations/instances_valTraffic.json'
    target_file = json.load(open(test, 'r'))

    # Make final dictionary
    dataset = dict.fromkeys(target_file.keys())
    dataset['info'] = target_file['info']
    dataset['licenses'] = target_file['licenses']
    dataset['categories'] = target_file['categories']
    dataset['annotations'] = anns
    dataset['images'] = imgs

    # Save to disk
    with open('../annotations/instances_'+str(filename)+'.json', 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=4)

    print('Saved dataset {} to disk!'.format(filename))

if __name__ == "__main__":
    cat_show = ['traffic light']  # Categories that you want shown and relabelled

    dataDir = "."
    dataType = "valTraffic"
    annFile = "{}/annotations/instances_{}.json".format(dataDir, dataType)

    imgDir = "images/valTraffic/"

    coco=COCO(annFile)

    cats = coco.loadCats(coco.getCatIds())
    nms = [cat['name'] for cat in cats]

    # Load image Ids
    catIds = coco.getCatIds(catNms=nms)
    imgIdsAll = coco.getImgIds()
    imgIds = imgIdsAll
    print('Number of images: ' + str(len(imgIds)))

    # Load annotations
    annIds = coco.getAnnIds(imgIds)
    anns = coco.loadAnns(annIds)
    print("Number of annotations: " + str(len(anns)))

    # Show image
    annId_i = 0
    go_backwards = False

    image = cv.imread("images/valTraffic/000000551647.jpg")
    print(image.shape)
    
    while annId_i < len(annIds):

        if annId_i < 0:
            print("You have reached the beginning")
            annId_i = 0
            go_backwards = False

        ann = anns[annId_i]
        imgId = ann['image_id']

        if coco.loadCats(ann['category_id'])[0]['name'] in cat_show:
            go_backwards = False
            
            image = cv.imread(imgDir + (str(imgId)+'.jpg').zfill(16))
            if image is None:
                raise Exception("Error: Cannot find image {}".format(imgDir + (str(imgId)+'.jpg').zfill(16)))
            
            b = ann['bbox']
            
            # Convert bounding boxes to (x1, y1, x2, y2)
            box = box_cxcywh_to_xyxy(b)
            print(coco.loadCats(ann['category_id'])[0]['name'])

            # Display bounding box
            image_bboxed = cv.rectangle(image, (box[0], box[1]), (box[2], box[3]), (252, 3, 219), 2)
            cv.imshow((str(imgId)+'.jpg'), image)
            if cv.waitKey(1) == ord("q"):
                break
            
            # Ask user for command
            while True:
                inp = str(input("Input q to quit, z to go backwards, and nothing to skip\n")).rstrip().lower()
                if inp == "":
                    break
                elif inp == "q":
                    exit()
                elif inp == "z":
                    go_backwards = True
                    break
                elif inp == "save":
                    break
                elif (inp == "r") or (inp == "1"):
                    print("r")
                    break
                elif (inp == "g") or (inp == "2"):
                    print("g")
                    break
                elif (inp == "n") or (inp == "3"):
                    print("na")
                    break
                else:
                    print("Invalid command")

            cv.destroyAllWindows()


        # Update image and annotation indices
        if go_backwards == False:
            annId_i += 1
        else:
            annId_i -= 1

    print("Completed image labelling")

