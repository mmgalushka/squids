# Explore TFRecords

This process helps to explore a collection of TFRecords as well as each record individually. It can be initiated by the Python `explore_tfrecords(...)` function (see [PyDoc](#pydoc) for more information)  or via a command line.

## Usage

<!-- Usage tab (Python|Shell)  -->

=== "Python"
    ```py
    from squids import explore_tfrecords
    explore_tfrecords(...)

    ```

    !!! Note
        If the `image_id` argument is not defined this function generates a list with summaries for all TFRecords, otherwise, it generates a summary and reconstructs an image (with overlays for categories, bounding boxes, and masks) for the specified record.

        This function works in two modes. If the `return_artifacts` argument is `True`, this function returns the collected information. If it is `False`, the collected information is printed to the console and stored in the file (in the specified `output_dir`).
=== "Shell"
    ```shell
    ~$ python squids.main explore [-h] [--no-categories] [--no-bboxes] [--no-segmentations] TFRECORDS_DIR [IMAGE_ID] [OUTPUT_DIR]

    positional arguments:
        TFRECORDS_DIR       a TFRecords directory to explore

    optional arguments:
        -h, --help          show this help message and exit

    A record exploration options:
        IMAGE_ID            an image ID to select
        OUTPUT_DIR          an output directory to save rendered image
        --no-categories     turn off showing of categories
        --no-bboxes         turn off showing of bounding boxes
        --no-segmentations  turn off showing of segmentations
    ```
    !!! Note
        If the `IMAGE_ID` is not defined this command lists summaries for all TFRecords, otherwise it outputs the specified record summary and stores the reconstructed image (with overlays for categories, bounding boxes, and masks) to the `OUTPUT_DIR`.

## Outcome

### Exploring All Records

The result of using this command for exploring all TFRecords is shown below.

    ```text
    dataset/synthetic-tfrecords/instances_train
    725 {1}      891 {1,2,3}  421 {1,3}    580 {2}      29 {3}       194 {1}    
    726 {2}      892 {3}      422 {1,2}    581 {3}      30 {1,2}     195 {1,3}  
    727 {3}      893 {3}      423 {1,3}    582 {1}      33 {1}       196 {2}    
    728 {2,3}    894 {2}      424 {1,3}    583 {1,2,3}  34 {2}       197 {1}    
    729 {1,2}    895 {2,3}    426 {1}      584 {1}      36 {1}       198 {1,2}  
    730 {3}      896 {1,3}    427 {3}      585 {3}      37 {3}       199 {2}    
    731 {1,2}    897 {1}      428 {1}      586 {1,2}    38 {3}       200 {2}    
    733 {2}      898 {2}      429 {2,3}    587 {1}      39 {1}       201 {2,3}  
    734 {2,3}    899 {1}      431 {2,3}    590 {3}      40 {3}       202 {2,3}  
    735 {1,2}    900 {1}      432 {3}      591 {3}      42 {1}       203 {1,3}  
    737 {1,2,3}  902 {3}      433 {1,3}    592 {3}      43 {1,2}     205 {2,3}  
    739 {2}      904 {2,3}    434 {2}      593 {1}      44 {1,2}     206 {3}    
    740 {2}      905 {2,3}    435 {3}      596 {3}      45 {2,3}     207 {2}    
    741 {1,2}    906 {1,2}    436 {2}      597 {3}      46 {1}       209 {1,2}  
    743 {2}      908 {1,3}    438 {3}      599 {2}      47 {1,3}     210 {1,3}  
    744 {2}      910 {2,3}    439 {3}      600 {2,3}    49 {1,2}     211 {2,3}  
    745 {1}      911 {1,2}    440 {1}      602 {3}      50 {1,2,3}   213 {2}    
    746 {2,3}    912 {1,2}    443 {2,3}    603 {1}      51 {1,2}     214 {2}    
    747 {1,2}    913 {3}      444 {3}      604 {3}      54 {3}       215 {1}    
    748 {1,3}    914 {2,3}    445 {1,3}    605 {3}      55 {3}       217 {2}    
    749 {1,2}    915 {1,3}    446 {2,3}    607 {3}      56 {3}       218 {2,3}  
    750 {1,2}    917 {1,2}    447 {1}      608 {1}      58 {2,3}     219 {1}    
    751 {2,3}    918 {1,2}    449 {1,2,3}  611 {1,2}    59 {1,2,3}   220 {3}
    ...          ...          ...          ...          ...          ...
    886 {1,2}    416 {2,3}    576 {1,2}    23 {3}       189 {1,2}    366 {1}    
    887 {1,3}    417 {1,3}    577 {2}      24 {2,3}     190 {2,3}    367 {1}    
    888 {1}      419 {2,3}    578 {1,2}    27 {2}       192 {1,3}  
    890 {1,3}    420 {1}      579 {2}      28 {3}       193 {1,2}  
    Total 715 records
    ```

From the output, you can observe the total number of images contained with these records, the listing of all images IDs combined with indicators categories present for each image. For example, if we have the following line `737 {1,2,3}`, it means that the image with ID `123` has 1 or more objects of category `1`, 1 or more objects of category `2` and 1 or more objects of category `3` respectively. This should help you pick a specific image including a specific set of categories.

!!! Note
    If you use  function `explore_tfrecords` with `return_artifacts==True` it returns a list of record IDs `[..., 737, 739, 740, 741, ...]` and summaries such as `[..., "{1,2,3}", "{2}", "{2}", "{1,2}", ...]`. You can apply your own style to its visualization.

### Exploring Individual Record

The individual record exploration produces the following output to the console and saves the image file `<image_id>.png` to the output directory.

    ```text
    Property                  Value
    ------------------------  -----------
    Image ID                  737
    Image Shape               (64, 64, 3)
    Total Labeled Objects     2
    Available Categories Set  {1, 2, 3}
    Image saved to ./737.png
    ```

The output contains information about the image identifier its shape, number of annotated objects, and their categories. Also, all information about bounding boxes, segmentation, and categories are overlaid to the image, which example is shown below.

![image with overlays](images/image_with_overlays.png)

!!! Note
    If you use  function `explore_tfrecords` with `return_artifacts==True` it returns two arguments: the first is a PIL image with overlays of categories, bounding boxes and masks and the second is a dictionary with the record summaries such as `{"image_id": 757, "image_shape": "(64, 64, 3)", ...}`.

## PyDoc

::: squids.tfrecords.explorer
    selection:
      members:
        - explore_tfrecords
