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
from itertools import groupby

import numpy as np


def mask_to_rle(binary_mask):
    """
        Run Length encode binary mask.

        Input:
            binary_mask: The binary mask. Expected input shape(h, w, n), where n is channels.

        Output:
            rles: List of run length encoded masks of size n.

    """
    h, w, n = binary_mask.shape
    rles = []
    for i in range(n):
        rle = {'counts': [], 'size': [h, w]}
        counts = rle.get('counts')
        for i, (value,
                elements) in enumerate(groupby(binary_mask.ravel(order='F'))):
            if i == 0 and value == 1:
                counts.append(0)
            counts.append(len(list(elements)))
        rles.append(rle)
    return rles


def rle_to_mask_v2(rle):
    """
        Convert RLE to binary mask

        Input:
            rle: Run length encoded binary masks of size n.

        Output:
            mask: Binary mask of shape (h, w).
    """
    h, w = rle['size']
    segm = rle['counts']
    k = []
    mask = np.zeros((w, h), dtype=np.uint8)
    b = 1
    c = 0
    for value in segm:
        if b:
            c += value
            b = 0
        else:
            b = 1
            i = c // h
            j = c % h
            while value:
                mask[i, j] = 1
                j += 1
                if j >= h:
                    j = 0
                    i += 1
                value -= 1
                c += 1
    return mask.T


def rle_to_mask(rle):
    """
        Convert RLE to binary mask

        Input:
            rle: Run length encoded binary masks of size n.

        Output:
            mask: Binary mask of shape (h, w).
    """
    h, w = rle['size']
    mask = np.zeros((w, h), dtype=np.uint8)
    c = 0
    b = 0
    for value in rle['counts']:
        end = c + value
        if b:
            start_x, start_y = c // h, c % h
            end_x, end_y = end // h, end % h

            while start_x < end_x:
                mask[start_x, start_y:] = 1
                start_y = 0
                start_x += 1

            start_x = min(start_x, w - 1)
            mask[start_x, start_y:end_y] = 1
        b = int(not b)
        c += value

    return mask.T


def rles_to_mask(rles):
    """
        Convert RLEs to binary mask

        Input:
            rles: list of run length encoded binary masks of size n.

        Output:
            masks: Binary mask of shape (h, w, n).
    """
    no_rles = len(rles)
    if no_rles == 0:
        return []
    h, w = rle[0]['size']
    masks = np.zeros((w, h, no_rles), dtype=np.uint8)
    for i, rle in enumerate(rles):
        mask = rle_to_mask(rle)
        masks[:, :, i] = mask
    return masks


def rle_iou(rles1, rles2):
    n = len(rles1)
    m = len(rles2)
    bboxes1 = get_boxes(rles1)
    bboxes2 = get_boxes(rles2)

    overlaps = compute_overlaps(bboxes1, bboxes2)
    for i in range(n):
        for j in range(m):
            if overlaps[i][j] > 0:
                if rles1[i]['size'] != rles2[j]['size']:
                    overlaps[i][j] = -1
                    continue

    return overlaps


def compute_iou(box, boxes, box_area, boxes_area):
    """Calculates IoU of the given box with the array of the given boxes.

        Inputs:
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

        Inputs:
            boxes1: [N1, (x1, y1, w, h)].
            boxes2: [N2, (x1, y1, w, h)].

        Outputs:
            ious: [N1, N2]
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
    """Upsample and get discrete points densely along entire boundary

        Inputs:
            xy: List of segmentation polygon. [x1, y1, x2, y2, x3, y3]
            k: size of the list
            h: Height of the mask
            w: Width of the mask
            scale: Density scale.

        Outputs:
            rle: Dictionary of run length encoded mask and image size.

    """
    k = len(xy) // 2
    x = [int(scale * xy[j] + 0.5) for j in range(0, 2 * k, 2)]
    y = [int(scale * xy[j] + 0.5) for j in range(1, 2 * k, 2)]
    x.append(x[0])
    y.append(y[0])

    u = [None] * m
    v = [None] * m

    m = 0
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
                u[m] = t + xs
                v[m] = int(ys + s * t + 0.5)
                m += 1
        else:
            for d in range(0, dy + 1):
                t = dy - d if flip else d
                v[m] = t + ys
                u[m] = int(xs + s * t + 0.5)
                m += 1

    k = len(u)
    x = []
    y = []
    for j in range(1, len(u)):
        if u[j] != u[j - 1]:
            xd = float(u[j] if u[j] < u[j - 1] else u[j] - 1)
            xd = ((xd + .5) / scale) - .5
            if int(xd) != xd or xd < 0 or xd > w - 1:
                continue

            yd = float(v[j] if v[j] < v[j - 1] else v[j - 1])
            yd = ((yd + .5) / scale) - .5

            if yd < 0:
                yd = 0
            elif yd > h:
                yd = h
            yd = int(yd)
            x[m] = int(xd)
            y[m] = yd
            m += 1

    if m == 0:
        x = []
        y = []
    else:
        x = x[:m]
        y = y[:m]

    k = len(x)
    a = [int(x[j] * int(h) + y[j]) for j in range(k)] + [int(h * w)]
    a.sort()
    k += 1

    p = 0
    for j in range(k):
        t = a[j]
        a[j] -= p
        p = t

    b = [a[0]]
    j = 1
    while j < k:
        if a[j] > 0:
            b[jb] = a[j]
            jb += 1
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
        Inputs:
            boxes: [N, (x1, y1, w, h)].

        Outputs:
            area: area of the bounding boxes. [N]
    """
    return boxes[:, 2] * boxes[:, 3]


def seg_area(segm):
    """
        Find the area of the Run length encoded mask. Every odd ind represents the 1 and even ind represents 0.
        We are adding all the counts at even numbers.
        [1, 1, 1, 0, 0, 1, 1] --> [0, 3, 2, 2]
        [0, 1, 1, 1, 0, 0, 1] --> [1, 3, 2, 1]

        Inputs:
            segm: Run length encoded(list) mask.

        Returns:
            area: the area of the mask
    """
    area = 0
    for ind, value in enumerate(segm):
        if ind % 2 == 1:
            area += value
    return area


def rles_area(rles):
    """Find the area of the Run length encoded masks.

        Inputs:
            rles: list of Run length encoded(list) mask.

        Returns:
            area: list of area of the mask
    """
    areas = []
    for _, rle in enumerate(rles):
        areas.append(seg_area(rle['counts']))
    return areas


def get_boxes(rles):
    """Get the bounding boxes of the Run length encoded masks.

        Inputs:
            rles: list of Run length encoded(list) mask.

        Returns:
            boxes: array of bounding boxes.
    """
    boxes = []
    for rle in rles:
        boxes.append(get_box(rle))
    boxes = np.array(boxes)
    return boxes


def get_box(rle):
    """Find the bounding box of the Run length encoded mask.

        Inputs:
            rle:  Run length encoded mask.

        Returns:
            box: bounding box.
    """
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
    """Find the bounding box of the segmentation polygon.

        Inputs:
            segm: list/single segmentation polygons.
            h: height of the image
            w: width of the image

        Returns:
            box: list of bounding boxes.
    """
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


def seg_to_rle(segm, h, w):
    """Convert the segmentation polygon to run length encoded mask.

        Inputs:
            segm: list/single segmentation polygons.
            h: height of the image
            w: width of the image

        Returns:
            rles: list of run length encoded masks.
    """
    rles = []
    if len(segm) == 0:
        return [{'counts': [], 'size': [0, 0]}]

    if type(segm[0]) == int:
        polygons = [segm]
    else:
        polygons = segm

    no_of_polygons = len(polygons)
    rles = [None] * no_of_polygons
    for i, poly in enumerate(polygons):
        k = len(poly) // 2
        rles[i] = rle_poly(poly, k, h, w)

    rle_mask = rle_merge(rles)[0]
    return rle_mask


def rleToBbox(rle):
    """Convert run length encoded mask to bounding box.

        Inputs:
            rle: run length encoded mask.

        Returns:
            bounding_box: bounding box.
    """
    h, w = rle['size']
    cnts = rle['counts']
    m = len(cnts)

    xs = w
    ys = h
    xe = 0
    ye = 0
    cc = 0

    if m == 0:
        bounding_box = [0, 0, 0, 0]

    for j, value in enumerate(cnts):
        if j > m - 2:
            break
        cc += value

        t = cc - j % 2
        y = t % h
        x = (t - y) / h

        if j % 2 == 0:
            xp = x
        elif xp < x:
            ys = 0
            ye = h - 1

        xs = min(xs, x)
        xe = max(xe, x)
        ys = min(ys, y)
        ye = max(ye, y)
    bounding_box = [xs, ys, xe - xs + 1, ye - ys + 1]
    return bounding_box


def rlesToBbox(rles):
    """Convert run length encoded masks to bounding boxes.

        Inputs:
            rle: list of run length encoded mask.

        Returns:
            bounding_box: list of bounding boxes.
    """
    bounding_box = []
    for rle in rles:
        bounding_box.append(rleToBbox(rle))

    return bounding_box


def rle_merge(rles, intersect=False):
    """Merge list of run length encoded masks to single rle.

        Inputs:
            rles: list of run length encoded masks.
            intersect: whether to apply intersection between two masks

        Returns:
            rles: run length encoded mask.
    """
    n = len(rles)
    if n <= 1:
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
