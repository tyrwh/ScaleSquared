import marimo

__generated_with = "0.10.16"
app = marimo.App(width="medium")


@app.cell
def _():
    import numpy as np
    from numpy import fft
    from matplotlib import pyplot as plt

    from skimage import io
    from scipy.signal import find_peaks
    from scipy.stats import variation
    from skimage.feature import peak_local_max
    from sklearn.cluster import DBSCAN

    return DBSCAN, fft, find_peaks, io, np, peak_local_max, plt, variation


@app.cell
def _(np):
    def rect_to_polar(nd: np.ndarray):
        if nd.ndim != 2:
            raise Exception("target array must be 2-dimensional\nshape of array supplied: %s" % nd.shape)
        if nd.shape[1] != 2:
            raise Exception("target array must be of dimension (n,2)\nshape of array supplied: %s" % nd.shape)
        r = np.sum(nd ** 2, 1) ** 0.5
        theta = np.array([np.arctan2(y,x) for y,x in nd])
        return (r, theta)
    return (rect_to_polar,)


@app.cell
def _(np):
    def find_base_radius(radii, max_error = 0.1):
        _sorted = np.sort(radii)
        _best_n = 1
        _best_i = None
        for _i in range(len(_sorted)):
            _scaled = _sorted / _sorted[_i]
            _errors = np.abs(_scaled - np.round(_scaled))
            if sum(_errors < max_error) > _best_n:
                _best_n = sum(_errors < max_error)
                _best_i = _i
        if _best_n < len(radii)/2:
            print('Could not sort at least half of radii into multiples of base number')
            return None
        _scaled = _sorted / _sorted[_best_i]
        _errors = np.abs(_scaled - np.round(_scaled))
        _multiples = radii[_errors < max_error]
        _multiples = _multiples / np.round(_scaled[_errors < max_error])
        
        return np.mean(_multiples)
    return (find_base_radius,)


@app.cell
def _(DBSCAN, fft, find_base_radius, np, peak_local_max, variation):
    def grid_fft(img, max_cv_r = 0.01, max_cv_wl = 0.01, min_peak_dist = 20, k_top=8):
        f_transform = fft.fft2(img)
        f_shift = fft.fftshift(f_transform)
        magnitude_2d = np.log(np.abs(f_shift))
        peak_coords = peak_local_max(magnitude_2d, min_distance=min_peak_dist)
        top_peaks = peak_coords[1:(2*k_top+1):2]
        centered = top_peaks - peak_coords[0]
        # transform into polar coords
        r = np.array(np.sum(centered ** 2, 1) ** 0.5)
        theta = np.array([np.arctan2(y,x) for y,x in centered])
        # make angles positive for convenience
        abs_theta = theta.copy()
        abs_theta[abs_theta < 0] += np.pi
        # cluster the angles and pull two largest clusters
        db = DBSCAN(eps=0.1, min_samples=int(k_top/4)).fit(abs_theta.reshape(-1,1))
        theta0 = np.mean(abs_theta[db.labels_ == 0])
        theta1 = np.mean(abs_theta[db.labels_ == 1])    
        # deviation from right angle should probably be < 0.05 radians
        dev_from_right_angle = abs((theta0-theta1)) / (np.pi/2) - 1
        if dev_from_right_angle > 0.05:
            raise Warning("Angles of two largest FFT peak clusters are > 0.05 radians from orthogonal. Check that grid is oriented correctly or try lowering k_top")
        # pull the radii
        r0_list = np.sort(r[db.labels_ == 0])
        r1_list = np.sort(r[db.labels_ == 1])
        # get polar coords of "average" peak distance, convert to rectangular
        polar_0 = (find_base_radius(r0_list), theta0)
        polar_1 = (find_base_radius(r1_list), theta1)
        rect_0 = np.array((polar_0[0] * np.sin(polar_0[1]), polar_0[0] * np.cos(polar_0[1])))
        rect_1 = np.array((polar_1[0] * np.sin(polar_1[1]), polar_1[0] * np.cos(polar_1[1])))
        # convert back into original shape and invert from frequency to wavelength in px
        # need to scale the dimensions of frequency based on the fact that the image isn't square
        min_side = np.min(img.shape)
        scaling_factor = img.shape / min_side
        wl0 = min_side / np.sqrt(np.sum((rect_0 / scaling_factor)**2))
        wl1 = min_side / np.sqrt(np.sum((rect_1 / scaling_factor)**2))
        if variation((wl0,wl1)) > max_cv_wl:
            print("Detected wavelengths have greater CV than tolerance: %s, %s" % (wl0,wl1))
            return None
        return np.sum((wl0,wl1))/2
    return (grid_fft,)


@app.cell
def _(io, plt):
    # some sample images

    blue = io.imread('sample_grids/blue_grid.jpg')[:,:,2]
    plt.figure(figsize=(9, 6))
    plt.imshow(blue, cmap='grey')
    return (blue,)


@app.cell
def _(blue, grid_fft):
    # need to set the min_peak_dist lower if the task fails
    print(grid_fft(blue, min_peak_dist=5))
    return


@app.cell
def _(io, plt):
    greenmat = io.imread('sample_grids/green_cutting_mat.jpg')[:,:,2]
    plt.figure(figsize=(9, 6))
    plt.imshow(greenmat, cmap='grey')
    return (greenmat,)


@app.cell
def _(greenmat, grid_fft):
    # need to set the min_peak_dist lower if the task fails
    print(grid_fft(greenmat))
    return


@app.cell
def _(io, plt):
    angled = io.imread('sample_grids/cutting_mat_angle.jpeg')[:,:,2]
    plt.figure(figsize=(9, 6))
    plt.imshow(angled, cmap='grey')
    return (angled,)


@app.cell
def _(angled, grid_fft):
    # the angled one finds the small grid squares first
    print(grid_fft(angled))

    print(grid_fft(angled, k_top=64))
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
