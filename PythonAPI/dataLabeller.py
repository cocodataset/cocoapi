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
# Modify label category
# Save modified annotations
# Save processed images
# Save weird images

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
    dataType = "val_custom"
    annFile = "{}/annotations/instances_{}.json".format(dataDir, dataType)
    coco=COCO(annFile)

    cats = coco.loadCats(coco.getCatIds())
    nms = [cat['name'] for cat in cats]

    catIds = coco.getCatIds(catNms=nms)
    imgIdsAll = coco.getImgIds()
    imgIds = imgIdsAll

    # Show image
    imgId_i = 0
    annId_i = 0
    go_backwards = False
    
    while (imgId_i < len(imgIds)) and (imgId_i > -1*len(imgIds)):
        imgId = imgIds[imgId_i]
        annIds = coco.getAnnIds(imgId)
        anns = coco.loadAnns(annIds)

        if len(anns) > 0:
            img = coco.loadImgs(imgId)
            ann = anns[annId_i]
            if coco.loadCats(ann['category_id'])[0]['name'] in cat_show:
                go_backwards = False
                
                image = cv.imread('%s/images/%s/%s'%(dataDir, dataType, img[0]['file_name']))
                
                b = ann['bbox']
                
                # Convert bounding boxes to (x1, y1, x2, y2)
                box = box_cxcywh_to_xyxy(b)
                print(coco.loadCats(ann['category_id'])[0]['name'])

                # Display bounding box
                image_bboxed = cv.rectangle(image, (box[0], box[1]), (box[2], box[3]), (252, 3, 219), 2)
                cv.imshow(img[0]['file_name'], image)
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
                    elif inp == "r":
                        print("r")
                        break
                    else:
                        print("Invalid command")


        # Update image and annotation indices
        if go_backwards == True:
            annId_i -= 1
            if (annId_i < 0):
                cv.destroyAllWindows()
                imgId_i -= 1
                
                imgId = imgIds[imgId_i]
                annIds = coco.getAnnIds(imgId)
                anns = coco.loadAnns(annIds)
                annId_i = len(anns) - 1
        else:
            annId_i += 1

            if (annId_i >= len(annIds)):
                cv.destroyAllWindows()
                annId_i = 0
                imgId_i += 1

