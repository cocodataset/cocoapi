__author__ = 'tsungyi'

# Interface for manipulating masks stored in RLE format.
#
# RLE is a simple yet efficient format for storing binary masks. RLE
# first divides a vector (or vectorized image) into a series of piecewise
# constant regions and then for each piece simply stores the length of
# that piece. For example, given M=[0 0 1 1 1 0 1] the RLE counts would
# be [2 3 1 1], or for M=[1 1 1 1 1 1 0] the counts would be [0 6 1]
# (note that the odd counts are always the numbers of zeros). Instead of
# storing the counts directly, additional compression is achieved with a
# variable bitrate representation based on a common scheme called LEB128.
#
# Compression is greatest given large piecewise constant regions.
# Specifically, the size of the RLE is proportional to the number of
# *boundaries* in M (or for an image the number of boundaries in the y
# direction). Assuming fairly simple shapes, the RLE representation is
# O(sqrt(n)) where n is number of pixels in the object. Hence space usage
# is substantially lower, especially for large simple objects (large n).
#
# Many common operations on masks can be computed directly using the RLE
# (without need for decoding). This includes computations such as area,
# union, intersection, etc. All of these operations are linear in the
# size of the RLE, in other words they are O(sqrt(n)) where n is the area
# of the object. Computing these operations on the original mask is O(n).
# Thus, using the RLE can result in substantial computational savings.
#
# The following API functions are defined:
#  encode         - Encode binary masks using RLE.
#  decode         - Decode binary masks encoded via RLE.
#  merge          - Compute union or intersection of encoded masks.
#  iou            - Compute intersection over union between masks.
#  area           - Compute area of encoded masks.
#  toBbox         - Get bounding boxes surrounding encoded masks.
#  frPyObjects    - Convert polygon, bbox, and uncompressed RLE to encoded RLE mask.
#
# Usage:
#  Rs     = encode( masks )
#  masks  = decode( Rs )
#  R      = merge( Rs, intersect=false )
#  o      = iou( dt, gt, iscrowd )
#  a      = area( Rs )
#  bbs    = toBbox( Rs )
#  Rs     = frPyObjects( [pyObjects], h, w )
#
# In the API the following formats are used:
#  Rs      - [dict] Run-length encoding of binary masks
#  R       - dict Run-length encoding of binary mask
#  masks   - [hxwxn] Binary mask(s) (must have type np.ndarray(dtype=uint8) in column-major order)
#  iscrowd - [nx1] list of np.ndarray. 1 indicates corresponding gt image has crowd region to ignore
#  bbs     - [nx4] Bounding box(es) stored as [x y w h]
#  poly    - Polygon stored as [[x1 y1 x2 y2...],[x1 y1 ...],...] (2D list)
#  dt,gt   - May be either bounding boxes or encoded masks
# Both poly and bbs are 0-indexed (bbox=[0 0 1 1] encloses first pixel).
#
# Finally, a note about the intersection over union (iou) computation.
# The standard iou of a ground truth (gt) and detected (dt) object is
#  iou(gt,dt) = area(intersect(gt,dt)) / area(union(gt,dt))
# For "crowd" regions, we use a modified criteria. If a gt object is
# marked as "iscrowd", we allow a dt to match any subregion of the gt.
# Choosing gt' in the crowd gt that best matches the dt can be done using
# gt'=intersect(dt,gt). Since by definition union(gt',dt)=dt, computing
#  iou(gt,dt,iscrowd) = iou(gt',dt) = area(intersect(gt,dt)) / area(dt)
# For crowd gt regions we use this modified criteria above for the iou.
#
# To compile run "python setup.py build_ext --inplace"
# Please do not contact us for help with compiling.
#
# Microsoft COCO Toolbox.      version 2.0
# Data, paper, and tutorials available at:  http://mscoco.org/
# Code written by Piotr Dollar and Tsung-Yi Lin, 2015.
# Licensed under the Simplified BSD License [see coco/license.txt]

import copy
import math

import numpy as np
import skimage


def compute_iou(box, boxes, box_area, boxes_area):
    """Calculates IoU of the given box with the array of the given boxes.
    box: 1D vector [x1, y1, w, h]
    boxes: [boxes_count, (x1, y1, w, h)]
    box_area: float. the area of 'box'
    boxes_area: array of length boxes_count.
    Note: the areas are passed in rather than calculated here for
    efficiency. Calculate once in the caller to avoid duplicate work.
    """
    # Calculate intersection areas
    y1 = np.maximum(box[1], boxes[:, 1])
    y2 = np.minimum(box[1] + box[3], boxes[:, 1] + boxes[:, 3])
    x1 = np.maximum(box[0], boxes[:, 0])
    x2 = np.minimum(box[0] + box[2], boxes[:, 0] + boxes[:, 2])
    intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    union = box_area + boxes_area[:] - intersection[:]
    iou = intersection / union
    return iou


def compute_overlaps(boxes1, boxes2):
    """Computes IoU overlaps between two sets of boxes.
    boxes1, boxes2: [N, (x1, y1, w, h)].
    For better performance, pass the largest set first and the smaller second.
    """
    if len(boxes1) == 0 or len(boxes2) == 0:
        return []

    # Areas of anchors and GT boxes
    area1 = get_box_area(boxes1)
    area2 = get_box_area(boxes2)

    # Compute overlaps to generate matrix [boxes1 count, boxes2 count]
    # Each cell contains the IoU value.
    overlaps = np.zeros((boxes1.shape[0], boxes2.shape[0]))
    for i in range(overlaps.shape[1]):
        box2 = boxes2[i]
        overlaps[:, i] = compute_iou(box2, boxes1, area2[i], area1)
    return overlaps


def rle_poly(xy, k, h, w, scale=5):
    x = []
    y = []
    for j in range(k):
        x.append(int(scale * xy[j * 2 + 0] + 0.5))
        y.append(int(scale * xy[j * 2 + 1] + 0.5))
    x.append(x[0])
    y.append(y[0])

    u = []
    v = []
    for j in range(k):
        xs, xe, ys, ye = x[j], x[j + 1], y[j], y[j + 1]
        dx, dy = abs(xe - xs), abs(ys - ye)
        if dx == 0 and dy == 0:
            continue
        flip = (dx >= dy and xs > xe) or (dx < dy and ys > ye)
        if flip:
            xs, xe = xe, xs
            ys, ye = ye, ys
        s = (ye - ys) / dx if dx >= dy else (xe - xs) / dy
        if dx >= dy:
            for d in range(0, dx + 1):
                t = dx - d if flip else d
                u.append(t + xs)
                v.append(int(ys + s * t + 0.5))
        else:
            for d in range(0, dy + 1):
                t = dy - d if flip else d
                v.append(t + ys)
                u.append(int(xs + s * t + 0.5))

    k = len(u)
    m = 0
    x = []
    y = []
    for j in range(1, len(u)):
        if u[j] != u[j - 1]:
            xd = float(u[j] if u[j] < u[j - 1] else u[j] - 1)
            xd = ((xd + .5) / scale) - .5
            if math.floor(xd) != xd or xd < 0 or xd > w - 1:
                continue

            yd = float(v[j] if v[j] < v[j - 1] else v[j - 1])
            yd = ((yd + .5) / scale) - .5

            if yd < 0:
                yd = 0
            elif yd > h:
                yd = h
            yd = math.ceil(yd)
            x.append(int(xd))
            y.append(int(yd))

    a = []
    k = len(x)
    for j in range(k):
        a.append(int(x[j] * int(h) + y[j]))
    a.append(int(h * w))
    a.sort()
    k += 1

    p = 0
    for j in range(k):
        t = a[j]
        a[j] -= p
        p = t

    j = 0
    b = []
    b.append(a[0])
    j += 1
    while j < k:
        if a[j] > 0:
            b.append(a[j])
            j += 1
        else:
            j += 1
            if j < k:
                b[-1] += a[j]
            j += 1

    rle = {'counts': b, 'size': [h, w]}
    return rle


def get_box_area(boxes):
    # Areas of anchors and GT boxes
    """Computes area of box.
    boxes: [N, (x1, y1, w, h)].
    """
    return boxes[:, 2] * boxes[:, 3]


def seg_mask(segm, height, width):
    mask = np.zeros([height, width], dtype=np.uint8)
    for seg in segm:
        seg = [round(i) - 1 for i in seg]
        poly = np.array(seg).reshape((int(len(seg) / 2), 2))
        rr, cc = skimage.draw.polygon(poly[:, 1], poly[:, 0])
        mask[rr, cc] = 1
    return mask


def seg_area(segm):
    area = 0
    for ind in range(1, len(segm), 2):
        area += segm[ind]
    return area


def get_box(rle):
    h, w = rle['size']
    rle_mask = rle['counts']
    m = int(len(rle_mask) / 2) * 2
    xs, ys, xe, ye, cc = w, h, 0, 0, 0
    if m == 0:
        box = [0, 0, 0, 0]

    for j in range(m):
        cc += rle_mask[j]
        t = cc - j % 2
        y = t % h
        x = (t - y) / h
        if j % 2 == 0:
            xp = x
        elif xp < x:
            ys = 0
            ye = h - 1
        xs, xe = min(xs, x), max(xe, x)
        ys, ye = min(ys, y), max(ye, y)

    box = [xs, ys, xe - xs + 1, ye - ys + 1]
    return box


def seg_to_box(segm, h, w):
    bbox = []
    if type(segm) == list:
        if type(segm[0]) == list:
            polygons = segm
        else:
            polygons = [segm]
        for polygon in polygons:
            k = len(polygon) // 2
            bbox.append(get_box(rle_poly(polygon, k, h, w)))
    else:
        bbox.append(get_box(segm))
    return bbox


def rle_to_mask(rle):
    h, w = rle['size']
    segm = rle['counts']
    mask = []
    for i, value in enumerate(segm):
        mask += [0] * value if i % 2 == 0 else [1] * value
    mask = np.array(mask)
    mask.resize(w, h)
    return mask.astype(np.uint8).T


def rle_merge(rles, intersect=False):
    n = len(rles)
    if n == 0:
        return rles
    elif n == 1:
        return rles

    cnts = rles[0]['counts']
    h, w = rles[0]['size']

    for i in range(1, n):
        B = rles[i]['counts']
        h1, w1 = rles[i]['size']
        if h1 != h or w1 != w:
            h = w = m = 0
            break

        A = copy.deepcopy(cnts)
        A_m = len(A)
        B_m = len(B)
        ca = A[0]
        cb = B[0]
        v = vb = va = 0
        m = 0
        a = b = 1
        cc = 0
        ct = 1
        while ct > 0:
            c = min(ca, cb)
            cc += c
            ct = 0
            ca -= c

            if ca == 0 and a < A_m:
                ca = A[a]
                a += 1
                va = int(not va)

            ct += ca
            cb -= c

            if cb == 0 and b < B_m:
                cb = B[b]
                b += 1
                vb = int(not vb)

            ct += cb
            vp = v

            if intersect:
                v = va and vb
            else:
                v = va or vb

            if v != vp or ct == 0:
                if m < len(cnts):
                    cnts[m] = cc
                    m += 1
                else:
                    cnts.append(cc)
                    m += 1
                cc = 0

    return [{'counts': cnts, 'size': rles[0]['size']}]
