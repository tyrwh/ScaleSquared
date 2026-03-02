import marimo

__generated_with = "0.18.0"
app = marimo.App(width="medium")


@app.cell
def _():
    from ultralytics import YOLO
    import skimage
    return YOLO, skimage


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Here is a short demo of trout morphometric segmentation. To run it, simply hit `Shift + Enter` to proceed through each cell
    """)
    return


@app.cell
def _(YOLO):
    # import the model itself
    # the model weights and relevant settings are all contained within this .pt file
    seg = YOLO('models/yolo11m_seg_full.pt')
    return (seg,)


@app.cell
def _(mo, skimage):
    # read in a sample image and visualize
    img = skimage.io.imread('sample_images/CX40F_10_25_2024_02.jpg')
    mo.image(img)
    return (img,)


@app.cell
def _(img, seg):
    # you can apply the model to an image with model.predict()
    seg.predict(img)
    return


@app.cell
def _(seg):
    # alternately, you can just provide a filepath to the image or dir
    # usually faster and simpler than the method above
    # but TIFFs can sometimes be wonky unless you read them in with PIL before running the model
    seg.predict('sample_images/CX40F_10_25_2024_02.jpg')
    return


@app.cell
def _(seg):
    # the output will always be a list of Results() objects, even if you only apply the model to a single image
    # we'll run it on all images in sample_images/
    seg_results = seg.predict('sample_images/')
    return (seg_results,)


@app.cell
def _(seg_results):
    # the Results object has some built-in methods for annotating
    # if you want custom colors, it may be easier to draw things yourself with OpenCV drawing functions
    seg_results[0].show()
    return


@app.cell
def _(seg_results):
    # similarly, you can save out the annotated images
    # NOTE - the underscore here is just a marimo convention
    for _res in seg_results:
        _res.save('results/' + _res.path.split('/')[-1])
    return


@app.cell
def _(seg_results):
    # the results themselves are contained within a Masks() object
    seg_results[0].masks
    return


@app.cell
def _(seg_results):
    # the classes are stored as a dict within .names
    # NOTE - this will contain all classes for the model, including those that were not found in the target image
    seg_results[0].names
    return


@app.cell
def _(seg_results):
    # masks.xy contains a list of ndarrays
    # each array contains of x-y coords in the original pixel coordinates of the input image
    # each item in the list corresponds to the respective class in the dict above
    seg_results[0].masks.xy
    return


@app.cell
def _(seg_results):
    # the coords in proportional terms are found in masks.xyn
    # useful if you are transforming or resizing
    seg_results[0].masks.xyn
    return


@app.cell
def _(seg_results):
    # binary masks are stored in masks.data
    # rather than a list of length n, it's stored as a single array of size n x H x W
    # note that H and W are the height and width of the resized image that was fed into the model, not the original image 
    seg_results[0].masks.data.shape
    return


@app.cell
def _(mo, seg_results):
    # let's plot and visualize a mask for reference
    body_mask = seg_results[0].masks.data[4,:,:]

    # we can see it's just 2-dimensional now, H x W
    print(body_mask.shape)

    # NOTE - the masks are stored as Tensors, not scalars
    # too much info to cover here, just know that you often need to convert back into a scalar to do certain operations
    mo.image(body_mask.numpy())
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
