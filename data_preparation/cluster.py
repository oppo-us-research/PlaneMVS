from sklearn.cluster import KMeans


def cluster(camera_planes, cluster_anchor_num=7):
    camera_normals = camera_planes[:, :3]

    kmeans_N = KMeans(n_clusters=cluster_anchor_num).fit(camera_normals)
    anchor_normals = kmeans_N.cluster_centers_

    return anchor_normals
