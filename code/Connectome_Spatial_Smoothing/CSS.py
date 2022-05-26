"""
This module provides the methods used in our paper to conduct Connectome Spatial
Smoothing (CSS).

The module contains the folowing functionalities:
    - Code to map high-resolution and atlas-resolution connetomes
    - Code to perform CSS

Python implementation of connectome-based smoothing
Author: Sina Mansour L.
Contact: sina.mansour.lakouraj@gmail.com
"""

import os
import datetime
import time
import sys
import itertools
import numpy as np
import scipy.sparse as sparse
import scipy.spatial as spatial
from scipy.interpolate import RegularGridInterpolator
import sklearn.preprocessing
import nibabel as nib
import gdist
import multiprocessing as mp


_main_dir = os.path.abspath(os.path.dirname(__file__))
_sample_cifti_dscalar = os.path.join(_main_dir, 'data/templates/cifti/ones.dscalar.nii')
_glasser_cifti = os.path.join(_main_dir, 'data/templates/atlas/Glasser360.32k_fs_LR.dlabel.nii')


def _join_path(*args):
    return os.path.join(*args)


def _write_sparse(sp_obj, file_path):
    sparse.save_npz(file_path, sp_obj)


def _load_sparse(file_path):
    return sparse.load_npz(file_path)


def _time_str(mode='abs', base=None):
    if mode == 'rel':
        return str(datetime.timedelta(seconds=(time.time() - base)))
    if mode == 'raw':
        return time.time()
    if mode == 'abs':
        return time.asctime(time.localtime(time.time()))


def _print_log(message, mode='info'):
    if mode == 'info':
        print ('{}: \033[0;32m[INFO]\033[0m {}'.format(_time_str(), message))
    if mode == 'err':
        print ('{}: \033[0;31m[ERROR]\033[0m {}'.format(_time_str(), message))
        quit()
    sys.stdout.flush()


def _handle_process_with_que(que, func, args, kwds):
    que.put(func(*args, **kwds))


def _get_sample_cifti_dscalar():
    # load sample scalar
    return nib.load(_sample_cifti_dscalar)


def _get_number_of_cortical_vertices_from_cifti_dscalar(cifti):
    # extract index counts from brain_models of cortical surfaces
    return np.sum(
        [
            x.index_count for x in cifti.header.get_index_map(1).brain_models
            if x.brain_structure in ['CIFTI_STRUCTURE_CORTEX_LEFT', 'CIFTI_STRUCTURE_CORTEX_RIGHT']
        ]
    )


def _max_smoothing_distance(sigma, epsilon, dim=2):
    """
    Computes the kernel radius as a function of kernel standard deviation and
    truncation threshold
    """
    return sigma * (-2 * np.log(epsilon)) ** (1 / dim)


def _diagonal_stack_sparse_matrices(m1, m2):
    """
    Inputs are expected to be CSR matrices
    this is what the output looks like:
    | M1  0 |
    | 0  M2 |
    """
    return sparse.vstack((
        sparse.hstack((
            m1,
            sparse.csr_matrix((m1.shape[0], m2.shape[1]), dtype=m1.dtype)
        )).tocsr(),
        sparse.hstack((
            sparse.csr_matrix((m2.shape[0], m1.shape[1]), dtype=m1.dtype),
            m2
        )).tocsr()
    ))


def _run_in_separate_process(func, *args, **kwds):
    que = mp.Queue()
    p = mp.Process(target=_handle_process_with_que, args=(que, func, args, kwds))
    p.start()
    # join removed as it caused the parent to sleep for no reason
    # p.join()
    return que.get()


def _sigma2fwhm(sigma):
    return sigma * np.sqrt(8 * np.log(2))


def _fwhm2sigma(fwhm):
    return fwhm / np.sqrt(8 * np.log(2))


# Handling parcellations


def _label_surface_atlas_to_data(cifti=_get_sample_cifti_dscalar(), atlas_file=_glasser_cifti, subcortex=False):
    """
    This function uses the hcp dlabel file of glasser's 360 cortical parcellation
    to generate a label list with the same size as the input cifti file and label
    it accordingly. (Glassers atlas is the default but can be changed)
    Note: the subcortical regions will be unlabeled as there isn't any
    information available about them in the original dlabel file.
    """
    # Load parcellation file
    parc = nib.load(atlas_file)

    #
    row_index = 0
    named_maps = [x for x in parc.header.get_index_map(0).named_maps][row_index]
    parc_dict = {x[0]: x[1].label for x in named_maps.label_table.items()}

    # a list of labels to fill in
    labels = ['Unlabeled'] * cifti.shape[1]

    # get Cortex labels from the dlabel file
    for structure_name in ['CIFTI_STRUCTURE_CORTEX_LEFT', 'CIFTI_STRUCTURE_CORTEX_RIGHT']:
        cifti_hem_cortex_brain_model = [x for x in cifti.header.get_index_map(1).brain_models if x.brain_structure == structure_name][0]
        parc_hem_cortex_brain_model = [x for x in parc.header.get_index_map(1).brain_models if x.brain_structure == structure_name][0]
        surface_to_label_index_dict = {x: i for i, x in enumerate(parc_hem_cortex_brain_model.vertex_indices)}
        for cifti_index, surface_index in enumerate(cifti_hem_cortex_brain_model.vertex_indices):
            try:
                label_index = surface_to_label_index_dict[surface_index] + parc_hem_cortex_brain_model.index_offset
                cifti_offset = cifti_hem_cortex_brain_model.index_offset
                labels[cifti_index + cifti_offset] = parc_dict[parc.get_fdata()[row_index, label_index]]
            except KeyError:
                continue

    # change ??? in gordon atlas to Unlabeled
    labels = [x if x not in ['???'] else 'Unlabeled' for x in labels]

    if subcortex:
        subcortical_structures = [x.brain_structure for x in cifti.header.get_index_map(1).brain_models][2:]

        for structure_name in subcortical_structures:
            brain_model = [x for x in cifti.header.get_index_map(1).brain_models if x.brain_structure == structure_name][0]
            for i in range(brain_model.index_count):
                labels[brain_model.index_offset + i] = structure_name

    return labels


def parcellation_characteristic_matrix(atlas_file=_glasser_cifti, cifti_file=_sample_cifti_dscalar, subcortex=False):
    '''
    This function generates a p x v characteristic matrix from a brain atlas.

    Args:

        atlas_file: path to a cifti atlas file (dlabel.nii) [default: HCP MMP1.0]

        cifti_template: [optional, default: HCP-YA template] a loaded dscalar cifti file (with nibabel.load), this template is used
                        to determine the high-resolution structure (number of cortical and subcortical
                        grayordinates).

        subcortex: [optional, default: False] boolean flag indicating whether subcortical labels are to be generated. If True, then
                   subcortical and cerebellar regions are labeled according to their brain model names in cifti format.

    Returns:

        parcellation_matrix: The p x v sparse matrix representation of the atlas.
    '''
    cifti_template = nib.load(cifti_file)
    surface_labels = _label_surface_atlas_to_data(cifti=cifti_template, atlas_file=atlas_file, subcortex=subcortex)

    cortical_vertices_count = _get_number_of_cortical_vertices_from_cifti_dscalar(cifti_template)

    # mask only cortical regions
    if not subcortex:
        surface_labels = surface_labels[:cortical_vertices_count]

    labels = [x for x in list(set(surface_labels)) if x != 'Unlabeled']
    labels.sort()

    label_dict = {x: i for (i, x) in enumerate(labels)}

    parcellation_matrix = np.zeros((len(labels), cortical_vertices_count))

    for (i, x) in enumerate(surface_labels):
        parcellation_matrix[label_dict[x], i] = 1

    return labels, sparse.csr_matrix(parcellation_matrix)


# CSS Smoothing kernel


def _local_geodesic_distances(max_distance, vertices, triangles):
    # distances = gdist.local_gdist_matrix(vertices.astype(np.float64), triangles.astype(np.int32), max_distance)
    distances = _run_in_separate_process(
        gdist.local_gdist_matrix,
        vertices.astype(np.float64),
        triangles.astype(np.int32),
        max_distance=max_distance,
    )

    # make sure maximum distance is applied
    distances[distances > max_distance] = 0
    distances = distances.minimum(distances.T)
    distances.eliminate_zeros()
    distances = distances.tocsr()
    return distances


def _local_geodesic_distances_on_surface(surface, max_distance):
    vertices = surface.darrays[0].data
    triangles = surface.darrays[1].data
    retval = _local_geodesic_distances(max_distance, vertices, triangles)
    return retval


def _trim_and_stack_local_distances(left_local_distances,
                                    right_local_distances,
                                    cifti_file):
    # load a sample file to read the mapping from
    cifti = nib.load(cifti_file)

    # load the brain models from the file (first two models are the left and right cortex)
    brain_models = [x for x in cifti.header.get_index_map(1).brain_models]

    # trim left surface to cortex
    left_cortex_model = brain_models[0]
    left_cortex_indices = left_cortex_model.vertex_indices[:]
    left_cortex_local_distance = left_local_distances[left_cortex_indices, :][:, left_cortex_indices]

    # trim right surface to cortex
    right_cortex_model = brain_models[1]
    right_cortex_indices = right_cortex_model.vertex_indices[:]
    right_cortex_local_distance = right_local_distances[right_cortex_indices, :][:, right_cortex_indices]

    # concatenate local distances with diagonal stacking
    return _diagonal_stack_sparse_matrices(left_cortex_local_distance, right_cortex_local_distance)


def _get_cortical_local_distances(left_surface_file, right_surface_file, max_distance, cifti_file):
    """
    This function computes the local distances on the cortical surface and returns a sparse matrix
    with dimensions equal to cortical brainordinates in the cifti file.
    """
    left_local_distances = _local_geodesic_distances_on_surface(nib.load(left_surface_file), max_distance)
    right_local_distances = _local_geodesic_distances_on_surface(nib.load(right_surface_file), max_distance)
    return _trim_and_stack_local_distances(left_local_distances, right_local_distances, cifti_file)


def _get_spatial_local_distances(xyz, max_distance):
    """
    This function computes the local distances of a list of 3d coordinates using a KD tree implementation.
    """
    kdtree = spatial.cKDTree(xyz)
    return kdtree.sparse_distance_matrix(kdtree, max_distance)


def _local_distances_to_smoothing_coefficients(local_distance, sigma):
    """
    Takes a sparse local distance symmetric matrix (CSR) as input,
    Generates an assymetric coefficient sparse matrix where each
    row i, has the coefficient for smoothing a signal from node i,
    therefore, each row sum is unit (1). sigma comes from the smoothing
    variance.
    """
    # apply gaussian transform

    gaussian = -(local_distance.power(2) / (2 * (sigma ** 2)))
    np.exp(gaussian.data, out=gaussian.data)

    # add ones to the diagonal
    gaussian += sparse.eye(gaussian.shape[0], dtype=gaussian.dtype).tocsr()

    # normalize rows of matrix
    return sklearn.preprocessing.normalize(gaussian, norm='l1', axis=0)


def compute_smoothing_kernel(left_surface_file, right_surface_file, fwhm, epsilon=0.01, cifti_file=_sample_cifti_dscalar, subcortex=False, volume_smoothing="integrated"):
    """
    Compute the CSS smoothing kernel on the cortical surfaces.

    Args:

        left_surface_file: The left hemisphere's surface in fs-LR 32k space (surf.gii format)

        right_surface_file: The right hemisphere's surface in fs-LR 32k space (surf.gii format)

        fwhm: The full width at half maximum (FWHM) in mm.

        epsilon: The kernel truncation threshold.

        cifti_file: [optional, default: HCP-YA template] path to a cifti file, this template is used
                    to determine the high-resolution structure to exclude the medial wall and potentially
                    integrate subcortical volume.

        subcortex: [optional, default: False] boolean flag indicating whether subcortical and cerebellar
                   voxels should also be include in the smoothing kernel. If True, then subcortical and
                   cerebellar regions are smoothed with a volumetric smoothing kernel.

        volume_smoothing: [optional, default: "integrated"] If subcortex is set to True, this input selects
                          the procedure to combine the volumetric smoothing with the cortical surface-based
                          (geodesic) smoothing. If set to "integrated" the smoothing kernel would allow for
                          smoothing weights between cortical surface nodes and the volumetric nodes according
                          to the Euclidean distance metric. If set to "independent" the volumetric and surface
                          smoothing kernels would be completely independent, i.e. the information will not be
                          smoothed across the volumetric and surface representations. The default option is
                          set to "integrated". this mainly allows for smoothing in between subcortical voxels
                          and insular vertices that are spatially proximal.

    Returns:

        kernel: The v x v high-resolution smoothing kernel.
    """
    sigma = _fwhm2sigma(fwhm)
    cortical_local_geodesic_distances = _get_cortical_local_distances(
        left_surface_file,
        right_surface_file,
        _max_smoothing_distance(sigma, epsilon),
        cifti_file,
    )
    if not subcortex:
        return _local_distances_to_smoothing_coefficients(
            cortical_local_geodesic_distances,
            sigma,
        )
    else:
        cortical_vertices_count = _get_number_of_cortical_vertices_from_cifti_dscalar(nib.load(cifti_file))
        if volume_smoothing == "independent":
            subcortical_local_euclidean_distances = _get_spatial_local_distances(
                _get_xyz(left_surface_file, right_surface_file, cifti_file=cifti_file)[cortical_vertices_count:],
                _max_smoothing_distance(
                    sigma,
                    epsilon,
                    2
                )
            )
            combined_local_distances = _diagonal_stack_sparse_matrices(
                cortical_local_geodesic_distances,
                subcortical_local_euclidean_distances
            )
        elif volume_smoothing == "integrated":
            volumetric_local_euclidean_distances = _get_spatial_local_distances(
                _get_xyz(left_surface_file, right_surface_file, cifti_file=cifti_file),
                _max_smoothing_distance(
                    sigma,
                    epsilon,
                    2
                )
            )
            combined_local_distances = sparse.bmat(
                [
                    [cortical_local_geodesic_distances, volumetric_local_euclidean_distances[:cortical_vertices_count, cortical_vertices_count:]],
                    [volumetric_local_euclidean_distances[cortical_vertices_count:, :cortical_vertices_count], volumetric_local_euclidean_distances[cortical_vertices_count:, cortical_vertices_count:]]
                ]
            )
        return _local_distances_to_smoothing_coefficients(
            combined_local_distances,
            sigma,
        )


def smooth_high_resolution_connectome(high_resolution_connectome, smoothing_kernel):
    """
    Perform CSS to smooth a high-resolution connectome.

    Args:

        high_resolution_connectome: The high-resolution structural connectome (v x v sparse CSR matrix)

        smoothing_kernel: The v x v CSS high-resolution smoothing kernel

    Returns:

        smoothed_high_resolution_connectome: The CSS smoothed connectome.
    """
    return smoothing_kernel.dot(high_resolution_connectome.dot(smoothing_kernel.T))


def smooth_parcellation_matrix(parcellation_matrix, smoothing_kernel):
    """
    Perform CSS to smooth a brain atlas parcellation (in the form of a p x v matrix) to a new soft parcellation..

    Args:

        parcellation_matrix: The p x v sparse matrix representation of the atlas.

        smoothing_kernel: The v x v CSS high-resolution smoothing kernel.

    Returns:

        smoothed_parcellation_matrix: The p x v matrix representation of the smoothed soft parcellation.
    """
    return parcellation_matrix.dot(smoothing_kernel)


# Map high-resolution structural connectome


def _get_xyz_hem_surface(hem_surface_file, brain_model_index, cifti_file):
    """
    returns the xyz mm coordinates of all brainordinates in that hemisphere's surface mesh (excludes medial wall)
    """
    img = nib.load(cifti_file)

    brain_models = [x for x in img.header.get_index_map(1).brain_models]

    hem_surface = nib.load(hem_surface_file)
    return hem_surface.darrays[0].data[brain_models[brain_model_index].vertex_indices]


def _get_xyz_surface(left_surface_file, right_surface_file, cifti_file):
    """
    returns the xyz mm coordinates of all brainordinates in the surface mesh (excludes medial wall)
    """
    # left cortex
    leftxyz = _get_xyz_hem_surface(left_surface_file, 0, cifti_file=cifti_file)

    # right cortex
    rightxyz = _get_xyz_hem_surface(right_surface_file, 1, cifti_file=cifti_file)

    return np.vstack([leftxyz, rightxyz])


def _get_xyz(left_surface_file, right_surface_file, cifti_file):
    """
    returns the xyz mm coordinates of all brainordinates in the cifti file (excludes medial wall, but includes subcortex)
    """
    img = nib.load(cifti_file)

    brain_models = [x for x in img.header.get_index_map(1).brain_models]

    # left cortex
    leftxyz = _get_xyz_hem_surface(left_surface_file, 0, cifti_file=cifti_file)

    # right cortex
    rightxyz = _get_xyz_hem_surface(right_surface_file, 1, cifti_file=cifti_file)

    # subcortical regions
    subijk = np.array(list(itertools.chain.from_iterable([(x.voxel_indices_ijk) for x in brain_models[2:]])))
    subxyz = nib.affines.apply_affine(img.header.get_index_map(1).volume.transformation_matrix_voxel_indices_ijk_to_xyz.matrix, subijk)

    xyz = np.vstack([leftxyz, rightxyz, subxyz])

    return xyz


def _apply_warp_to_points_mm_to_mm(native_mms, warp_file=None, warp=None):
    """
    This function is used to warp a list of points from one mm space to another mm space
    using a nonlinear warpfield file. make sure to put the reverse warp file (i.e. standard2acpc
    from the HCP files can warp the points from native to MNI)
    Note: points are given as a m*3 array.
    """
    # only read warp if not provided
    if warp is None:
        warp = nib.load(warp_file)

    # construct the 3-d grid
    x = np.linspace(0, warp.shape[0] - 1, warp.shape[0])
    y = np.linspace(0, warp.shape[1] - 1, warp.shape[1])
    z = np.linspace(0, warp.shape[2] - 1, warp.shape[2])

    # fit a nonlinear interpolator
    xinterpolate = RegularGridInterpolator((x, y, z), warp.get_fdata()[:, :, :, 0])
    yinterpolate = RegularGridInterpolator((x, y, z), warp.get_fdata()[:, :, :, 1])
    zinterpolate = RegularGridInterpolator((x, y, z), warp.get_fdata()[:, :, :, 2])

    # first run the linear transformation
    native_voxs = nib.affines.apply_affine(np.linalg.inv(warp.affine), native_mms)

    # next run the nonlinear transformation
    dx_mm, dy_mm, dz_mm = (-xinterpolate(native_voxs), yinterpolate(native_voxs), zinterpolate(native_voxs))

    return native_mms + np.array([dx_mm, dy_mm, dz_mm]).T


def save_warped_tractography(track_file,
                             warp_file,
                             out_file):
    """
    Reads in a .tck tractography file and uses a nonlinear warp to transform the tractogram
    to a new space. This can be used to warp the output of tractography from native space
    to a standard template space.

    NOTE: This only performs a nonlinear warp, any prior linear transformations should be
          applied. Normally, a transformation from native to MNI can include flirt + fnirt.
          While flirt performs an initial linear warp, fnirt generates a nonlinear warp to
          be used after the linear transformation. The original tractography may need a
          linear transformation first, in which case, MRtrix's tcktransform function could
          be used to compute the warped tractograms.

    Args:

        track_file: The tractography file to be warped (tck format)

        warp_file: The nonlinear warp to nonlinearly transform streamlines (xfms -> nii file.

        out_file: The path to save the warped tractography.

    """
    # load the track file streamlines
    tracks = nib.streamlines.load(track_file)

    # concatenate all coordinates into a single (Nx3) vector
    concatenated_coords = np.concatenate(tracks.streamlines)

    # warp the list of 3 dimensional coordinates
    warped_concatenated_coords = _apply_warp_to_points_mm_to_mm(concatenated_coords, warp=nib.load(warp_file))

    # now rewrap warped coordinates back to the tractogram
    start = 0
    for ind in range(len(tracks.streamlines)):
        step = tracks.streamlines[ind].shape[0]
        tracks.streamlines[ind] = warped_concatenated_coords[start:(start + step)]
        start += step

    # save the output
    tracks.save(out_file)

    return tracks


def _get_endpoint_distances_from_tractography(track_file,
                                              left_surface_file,
                                              right_surface_file,
                                              cifti_file,
                                              warp_file=None,
                                              subcortex=False,
                                              ):
    """
    Returns the streamline endpoint distances from closest vertex on cortical surface mesh
    and the closest vertex index. Additionally warps the endpoints before aligning to the
    surface mesh, if a warp file is provided. This is useful when mapping a native
    tractography file to a standard space surface mesh.
    """
    # load the track file streamlines
    _print_log('loading tractography file.')
    tracks = nib.streamlines.load(track_file)
    _print_log('track file loaded: {}'.format(track_file))

    # extract streamline endpoints
    starts = np.array([stream[0] for stream in tracks.streamlines])
    ends = np.array([stream[-1] for stream in tracks.streamlines])
    _print_log('endpoints extracted: #{}'.format(len(starts)))

    if warp_file is not None:
        # calculate endpoint coordinates in the MNI space
        warped_starts = _apply_warp_to_points_mm_to_mm(starts, warp_file)
        warped_ends = _apply_warp_to_points_mm_to_mm(ends, warp_file)
        _print_log('endpoints warped: #{}'.format(len(starts)))
    else:
        warped_starts = starts
        warped_ends = ends

    # extract cortical surface coordinates
    if subcortex:
        xyz = _get_xyz(left_surface_file, right_surface_file, cifti_file=cifti_file)
    else:
        xyz = _get_xyz_surface(left_surface_file, right_surface_file, cifti_file=cifti_file)

    # store the coordinates in a kd-tree data structure to locate closest point faster
    kdtree = spatial.cKDTree(xyz)

    # locate closest surface points to every endpoint
    start_dists, start_indices = kdtree.query(warped_starts)
    end_dists, end_indices = kdtree.query(warped_ends)
    _print_log('closest brainordinates located')

    return (start_dists, start_indices, end_dists, end_indices, len(xyz))


def _get_half_incidence_matrices_from_endpoint_distances(start_dists,
                                                         start_indices,
                                                         end_dists,
                                                         end_indices,
                                                         node_count,
                                                         threshold):
    """
    Returns two half incidence matrices in a sparse format (CSR) after
    filtering the streamlines that are far (>2mm) from their closest vertex.
    """
    # mask points that are further than the threshold from all surface coordinates
    outlier_mask = (start_dists > threshold) | (end_dists > threshold)
    _print_log('outliers located: #{} outliers ({}%, with threshold {}mm)'.format(
        sum(outlier_mask),
        (100 * sum(outlier_mask)) / len(outlier_mask),
        threshold,
    ))

    # create a sparse incidence matrix
    _print_log('creating sparse incidence matrix')
    start_dict = {}
    end_dict = {}
    indices = (i for i in range(len(outlier_mask)) if not outlier_mask[i])
    for l, i in enumerate(indices):
        start_dict[(start_indices[i], l)] = start_dict.get((start_indices[i], l), 0) + 1
        end_dict[(end_indices[i], l)] = end_dict.get((end_indices[i], l), 0) + 1

    start_inc_mat = sparse.dok_matrix(
        (
            node_count,
            (len(outlier_mask) - outlier_mask.sum())
        ),
        dtype=np.float32
    )

    for key in start_dict:
        start_inc_mat[key] = start_dict[key]

    end_inc_mat = sparse.dok_matrix(
        (
            node_count,
            (len(outlier_mask) - outlier_mask.sum())
        ),
        dtype=np.float32
    )

    for key in end_dict:
        end_inc_mat[key] = end_dict[key]

    _print_log('sparse matrix generated')

    return (start_inc_mat.tocsr(), end_inc_mat.tocsr())


def _get_adjacency_from_half_incidence_matrices(U, V):
    """
    return a sparse adjacency matrix A from the two halfs of incidence matrix U & V.
    """
    A = U.dot(V.T)
    return A + A.T


def map_high_resolution_structural_connectivity(track_file,
                                                left_surface_file,
                                                right_surface_file,
                                                warp_file=None,
                                                threshold=2,
                                                subcortex=False,
                                                cifti_file=_sample_cifti_dscalar,):
    """
    Map the high-resolution structural connectivity matrix from tractography outputs.

    Args:

        track_file: The tractography file to map connectivity from (tck format)

        left_surface_file: The left hemisphere's surface in fs-LR 32k space (surf.gii format)

        right_surface_file: The right hemisphere's surface in fs-LR 32k space (surf.gii format)

        warp_file: [optional] A nonlinear warp can be provided to map streamlines after warping
                   the endpoints.

        threshold: [default=2] A threshold to exclude endpoints further than that threshold from
                   any cortical vertices (in mm).

        subcortex: [optional, default: False] A flag to indicate whether subcortical and cerebellar
                   voxels should also be included in the high-resolution connectivity map.

        cifti_file: [optional, default: HCP-YA template] Path to a cifti file, this template is used
                    to determine the high-resolution structure to exclude the medial wall and potentially
                    integrate subcortical volume.

    Returns:

        connectome: The high-resolution structural connectome in a sparse csr format.
    """
    return _get_adjacency_from_half_incidence_matrices(
        *_get_half_incidence_matrices_from_endpoint_distances(
            *_get_endpoint_distances_from_tractography(
                track_file,
                left_surface_file,
                right_surface_file,
                cifti_file,
                warp_file,
                subcortex=subcortex,
            ),
            threshold=threshold
        )
    )


def downsample_high_resolution_structural_connectivity_to_atlas(high_resolution_connectome,
                                                                parcellation):
    """
    Downsample the high-resolution structural connectivity matrix to the resolution of a brain atlas.

    Args:

        high_resolution_connectome: The high-resolution structural connectome (v x v sparse CSR matrix)

        parcellation: A p x v sparse percellation matrix (can also accept a soft parcellation)

    Returns:

        connectome: The atlas-resolution structural connectome in a sparse csr format.
    """
    return parcellation.dot(high_resolution_connectome.dot(parcellation.T))


def map_atlas_resolution_structural_connectivity(track_file,
                                                 left_surface_file,
                                                 right_surface_file,
                                                 atlas_file=_glasser_cifti,
                                                 warp_file=None,
                                                 threshold=2,
                                                 subcortex=False,
                                                 cifti_file=_sample_cifti_dscalar,):
    """
    Maps the structural connectivity matrix at the resolution of a brain atlas.

    Args:

        track_file: The tractography file to map connectivity from (tck format)

        left_surface_file: The left hemisphere's surface in fs-LR 32k space (surf.gii format)

        right_surface_file: The right hemisphere's surface in fs-LR 32k space (surf.gii format)

        warp_file: [optional] A nonlinear warp can be provided to map streamlines after warping
                   the endpoints.

        threshold: [default=2] A threshold to exclude endpoints further than that threshold from
                   any cortical vertices (in mm).

        subcortex: [optional, default: False] A flag to indicate whether subcortical and cerebellar
                   voxels should be included and labeled by structure name.

        cifti_file: [optional, default: HCP-YA template] Path to a cifti file, this template is used
                    to determine the high-resolution structure to exclude the medial wall and potentially
                    integrate subcortical volume.

    Returns:

        connectome: The atlas-resolution structural connectome in a sparse csr format.
    """
    return downsample_high_resolution_structural_connectivity_to_atlas(
        map_high_resolution_structural_connectivity(
            track_file,
            left_surface_file,
            right_surface_file,
            warp_file=warp_file,
            threshold=threshold,
            subcortex=subcortex,
        ),
        parcellation_characteristic_matrix(
            atlas_file=atlas_file,
            cifti_file=cifti_file,
            subcortex=subcortex,
        )[1]
    )
