import numpy as np
import cv2


def nnf(img_a, img_b, patch_size, iterations):
    offsets, distances, img_padding = initialization(img_a, img_b, patch_size)
    offsets = iteration(offsets, distances, img_padding, img_a, img_b, patch_size, iterations)

    return offsets


def initialization(img_a, img_b, patch_size):
    h_a, w_a = img_a.shape[:2]
    h_b, w_b = img_b.shape[:2]

    offsets = np.empty([h_a, w_a], dtype=object)
    distances = np.empty([h_a, w_a])

    patch_radius = patch_size // 2

    random_b1 = np.random.randint(patch_radius, h_b-patch_radius, [h_a, w_a])
    random_b2 = np.random.randint(patch_radius, w_b-patch_radius, [h_a, w_a])

    padding_a = np.full([h_a+2*patch_radius, w_a+2*patch_radius, 3], np.nan)
    padding_a[patch_radius:h_a+patch_radius, patch_radius:w_a+patch_radius, :] = img_a

    for i in range(h_a):
        for j in range(w_a):
            a = np.array([i, j])
            b = np.array([random_b1[i, j], random_b2[i, j]])
            offsets[i, j] = b
            distances[i, j] = get_distance(a, b, padding_a, img_b, patch_size)

    return offsets, distances, padding_a


def iteration(offsets, distances, img_padding, img_a, img_b, patch_size, iterations):
    h, w = img_a.shape[:2]

    for i in range(iterations):
        if i % 2 == 0:
            for k in reversed(range(h)):
                for j in reversed(range(w)):
                    a = np.array([k, j])
                    propagation(offsets, a, distances, img_padding, img_b, patch_size, is_odd=False)
                    random_search(offsets, a, distances, img_padding, img_b, patch_size)
        else:
            for k in range(h):
                for j in range(w):
                    a = np.array([k, j])
                    propagation(offsets, a, distances, img_padding, img_b, patch_size, is_odd=True)
                    random_search(offsets, a, distances, img_padding, img_b, patch_size)
    return offsets


def get_distance(a, b, padding_a, img_b, patch_size):
    patch_radius = patch_size // 2

    patch_a = padding_a[a[0]:a[0]+patch_size, a[1]:a[1]+patch_size, :]
    patch_b = img_b[b[0]-patch_radius:b[0]+patch_radius+1, b[1]-patch_radius:b[1]+patch_radius+1, :]

    return np.sum(np.square(np.nan_to_num(patch_b - patch_a)))/np.sum(1 - np.isnan(patch_b - patch_a))


def propagation(offsets, a, distances, padding_img_a, img_b, patch_size, is_odd):
    h_a, w_a = np.array(padding_img_a.shape[:2]) - patch_size + 1
    x, y = a[:2]

    current_distance = distances[x, y]

    if is_odd:
        left_distance = distances[max(x-1, 0), y]
        up_distance = distances[x, max(y-1, 0)]

        min_distance_idx = np.argmin([current_distance, left_distance, up_distance])

        if min_distance_idx == 1:
            offsets[x, y] = offsets[max(x-1, 0), y]
            distances[x, y] = get_distance(a, offsets[x, y], padding_img_a, img_b, patch_size)
        elif min_distance_idx == 2:
            offsets[x, y] = offsets[x, max(y-1, 0)]
            distances[x, y] = get_distance(a, offsets[x, y], padding_img_a, img_b, patch_size)
    else:
        right_distance = distances[min(x+1, h_a-1), y]
        down_distance = distances[x, min(y+1, w_a-1)]

        min_distance_idx = np.argmin([current_distance, right_distance, down_distance])

        if min_distance_idx == 1:
            offsets[x, y] = offsets[min(x+1, h_a-1), y]
            distances[x, y] = get_distance(a, offsets[x, y], padding_img_a, img_b, patch_size)
        elif min_distance_idx == 2:
            offsets[x, y] = offsets[x, min(y+1, w_a-1)]
            distances[x, y] = get_distance(a, offsets[x, y], padding_img_a, img_b, patch_size)


def random_search(offsets, a, distances, padding_img_a, img_b, patch_size):
    x, y = a[:2]
    b_x, b_y = offsets[x, y][:2]
    h_b, w_b = img_b.shape[:2]

    patch_radius = patch_size // 2

    def search_dim(dim, i): return dim * np.power(0.5, i)

    i = 1
    search_h = search_dim(h_b, i)
    search_w = search_dim(w_b, i)

    while search_h > 1 and search_w > 1:
        random_b_x = np.random.randint(max(b_x-search_h, patch_radius), min(b_x+search_h, h_b-patch_radius))
        random_b_y = np.random.randint(max(b_y-search_w, patch_radius), min(b_y+search_w, w_b-patch_radius))

        search_h = search_dim(h_b, i)
        search_w = search_dim(w_b, i)

        b = np.array([random_b_x, random_b_y])
        distance_updated = get_distance(a, b, padding_img_a, img_b, patch_size)

        if distance_updated < distances[x, y]:
            distances[x, y] = distance_updated
            offsets[x, y] = b

        i += 1


img_a = cv2.imread("./v001.jpg")
img_b = cv2.imread("./v002.jpg")

# img_a = np.random.randint(0, 255, (5, 5, 3))
# img_b = np.random.randint(0, 255, (5, 5, 3))

offsets = nnf(img_a, img_b, patch_size=3, iterations=4)
print(offsets)
