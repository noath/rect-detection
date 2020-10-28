import cv2
import imutils
import matplotlib.pyplot as plt
import numpy as np
import os

from sys import argv


def preprocess_img(path_to_img):
    image = cv2.imread(path_to_img)
    resized = imutils.resize(image, width=300)
    ratio = image.shape[0] / float(resized.shape[0])

    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]

    return thresh, ratio


def find_contours(threshold_img):
    items = cv2.findContours(
        threshold_img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    hierarchy = items[1][0]
    cnts = imutils.grab_contours(items)
    return cnts, hierarchy


def detect_shape(c):
    ar = None
    shape = "unidentified"
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.05 * peri, True)
    if len(approx) == 3:
        shape = "triangle"
    elif len(approx) == 4:
        _, _, w, h = cv2.boundingRect(approx)
        ar = max(w, h) / min(w, h)
        shape = "rectangle"
    elif len(approx) == 5:
        shape = "pentagon"
    else:
        shape = "circle"
    return shape, ar


def find_child_contours(hierarchy, c_idx):
    return np.where(hierarchy[:, 3] == c_idx)[0]


def check_points_similarity(point1, point2, eps=3):
    if len(point1) == 1:
        point1 = point1[0]
    if len(point2) == 1:
        point2 = point2[0]
    return abs(point1[0] - point2[0]) <= eps and abs(point1[1] - point2[1]) <= eps


def find_similar_points_indices(sample, cnt, eps=3):
    # finds indices of sample's points
    # which lie close to any points from cnt
    sim_idx = set()
    for c_point in cnt:
        for idx, s_point in enumerate(sample):
            if check_points_similarity(c_point, s_point, eps):
                sim_idx.add(idx)
    return sim_idx


def check_polygons_similarity(poly1, poly2, eps=3, conf=10):
    sim_pts = find_similar_points_indices(poly1, poly2, eps)
    l1, l2 = len(poly1), len(poly2)
    return (
        l1 - l1 // conf <= len(sim_pts) <= l1 + l1 // conf
        or l2 - l2 // conf <= len(sim_pts) <= l2 + l2 // conf
    )


def collapse_similar_contours(cnts, eps=3, conf=10):
    final_idx = set(list(range(len(cnts))))
    for i, cnt1 in enumerate(cnts):
        for ii, cnt2 in enumerate(cnts[i + 1 :]):
            j = i + 1 + ii
            if check_polygons_similarity(cnt1, cnt2, eps, conf):
                peri1 = cv2.arcLength(cnt1, True)
                peri2 = cv2.arcLength(cnt2, True)
                if peri1 >= peri2:
                    final_idx.discard(i)
                else:
                    final_idx.discard(j)
    return final_idx


def try_overlapping_rectangle(cnt, aspect_ratio_lim=np.inf, delta=0.1):
    rect = cv2.minAreaRect(cnt)
    box = np.array(cv2.boxPoints(rect))
    diff_ratio = abs(1 - cv2.contourArea(cnt) / cv2.contourArea(box))
    ar = max(rect[1]) / min(rect[1])
    if ar <= aspect_ratio_lim and diff_ratio <= delta:
        return (box, ar)
    return None, None


def draw_rectangle_contours(path_to_img, ratio, cnts, write_ar=True, output_dir=None):
    detected = cv2.imread(path_to_img)
    origin = detected.copy()
    for c, ar in cnts:
        M = cv2.moments(c)
        cX = int((M["m10"] / M["m00"]) * ratio)
        cY = int((M["m01"] / M["m00"]) * ratio)
        c = c.astype("float")
        c *= ratio
        c = c.astype("int")
        cv2.drawContours(detected, [c], -1, (0, 255, 0), 3)

        if ar != np.inf and write_ar:
            ar_str = f"1:{round(ar, 2)}"
            cv2.putText(
                detected,
                ar_str,
                (cX, cY),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (255, 255, 255),
                3,
            )
    plt.figure(figsize=(20, 10))
    plt.subplot(1, 2, 1)
    plt.title("Origin image", fontsize=18)
    plt.imshow(origin)

    plt.subplot(1, 2, 2)
    plt.title("Detection result", fontsize=18)
    plt.imshow(detected)

    if output_dir is not None:
        fname = os.path.split(path_to_img)[-1]
        plt.savefig(os.path.join(output_dir, fname))
    else:
        plt.show()


def detect_rectangles(
    path_to_img,
    aspect_ratio_limit=3,
    area_limit=100,
    delta=0.15,
    smoothness_limit=90,
    eps=5,
    conf=10,
    draw=True,
    write_ar=True,
    output_dir=None,
):
    thresh, ratio = preprocess_img(path_to_img)
    cnts, hierarchy = find_contours(thresh)

    final_cnts = []
    for i, c in enumerate(cnts):
        # skipping small contours
        if abs(cv2.contourArea(c)) < area_limit:
            continue
        # try to find rectangles in child contours by set difference
        for j in find_child_contours(hierarchy, i):
            c_child = cnts[j]
            sim_idx = find_similar_points_indices(c, c_child, eps=eps)
            c_diff = np.array([point for i, point in enumerate(c) if i not in sim_idx])
            if (len(c_diff) >= len(c) // 3) or (
                len(c) >= smoothness_limit and len(c_diff) >= 4
            ):
                diff_shape, diff_ar = detect_shape(c_diff)
                if diff_shape == "rectangle" or diff_shape == "pentagon":
                    diff_box, diff_ar = try_overlapping_rectangle(
                        c_diff, aspect_ratio_limit, delta
                    )
                    if diff_box is not None and diff_ar is not None:
                        final_cnts.append((diff_box, diff_ar))

        shape, _ = detect_shape(c)
        if shape != "rectangle":
            continue
        box, ar = try_overlapping_rectangle(c, aspect_ratio_limit, delta)
        if box is not None and ar is not None:
            final_cnts.append((box, ar))

    # collapsing similar contours
    collapsed_idx = collapse_similar_contours(
        [c for c, ar in final_cnts], eps=eps, conf=conf
    )
    final_cnts = [cnt for idx, cnt in enumerate(final_cnts) if idx in collapsed_idx]

    if draw:
        draw_rectangle_contours(
            path_to_img, ratio, final_cnts, write_ar=write_ar, output_dir=output_dir
        )

    return final_cnts


if __name__ == "__main__":
    if len(argv) != 3:
        print(f"Usage: {argv[0]} input_img_path output_dir")
        exit(0)
    img_path, output_dir = argv[1:]
    detect_rectangles(img_path, output_dir=output_dir)
