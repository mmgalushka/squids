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
        If the `image_id` argument is:
            
        - defined - this command outputs the specified record summary and stores the reconstructed image (with overlays for categories, bounding boxes, and masks) to the `output_dir`;
        - not defined - this command lists summaries for all TFRecords.

        This function works in two modes. If the `return_artifacts` argument is:
        
        - `True`, this function returns the collected information; 
        - `False`, the collected information is printed to the console and stored in the file (in the specified `output_dir`).

    ## Outcome

    ### Exploring All Records

    The result of using this function for exploring all TFRecords is shown below.

    ```python
    from squids import explore_tfrecords

    record_ids, record_summaries = explore_tfrecords(
        "dataset/synthetic-tfrecords/instances_train",
        return_artifacts=True
    )

    print("record_ids:", record_ids)
    print("record_summaries:", record_summaries)
    ```

    ```text
    record_ids: [725, 726, 727, 728, 729, 730, 731, 733, 734, 735, 737, 739, 740, 741, 743, 744, 745, 746, 747, 748, 749, 750, 751, 753, 754, 756, 758, 759, 760, 761, 763, 765, 767, 768, 770, 771, 772, 773, 775, 776, 778, 782, 784, 785, 786, 787, 789, 790, 791, 793, 795, ...]

    record_summaries: [Counter({1: 2}), Counter({2: 1}), Counter({3: 2}), Counter({3: 1, 2: 1}), Counter({2: 2, 1: 1}), Counter({3: 2}), Counter({2: 2, 1: 1}), Counter({2: 1}), Counter({3: 2, 2: 1}), Counter({2: 1, 1: 1}), Counter({2: 1, 1: 1, 3: 1}), Counter({2: 2}), Counter({2: 3}), Counter({1: 1, 2: 1}), Counter({2: 1}), Counter({2: 1}), Counter({1: 1}), Counter({3: 1, 2: 1}), Counter({2: 1, 1: 1}), Counter({1: 2, 3: 1}), Counter({1: 1, 2: 1}), Counter({1: 2, 2: 1}), Counter({2: 1, 3: 1}), Counter({1: 1}), Counter({2: 2, 1: 1}), Counter({1: 3}), Counter({3: 1, 1: 1}), Counter({2: 1}), Counter({3: 1, 2: 1}), Counter({3: 2, 1: 1}), Counter({3: 1}), Counter({3: 1}), Counter({3: 1, 2: 1, 1: 1}), Counter({1: 1}), Counter({3: 2}), Counter({3: 1, 1: 1}), Counter({3: 1}), Counter({3: 2, 1: 1}), Counter({2: 1, 1: 1}), Counter({3: 2, 2: 1}), Counter({1: 2, 3: 1}), Counter({2: 1}), Counter({2: 1}), Counter({2: 1}), Counter({1: 2, 2: 1}), Counter({1: 2, 2: 1}), Counter({3: 2, 2: 1}), Counter({2: 1}), Counter({1: 2, 2: 1}), Counter({3: 1}), Counter({2: 1}), ...]
    ```

    From the output, you can observe two arrays. The `record_ids` array contains record (image) IDs present in the training dataset.  The `record_summaries` array contains `Counter` Python objects for each record (image) with information of what categories are present in the images and how many objects belonged to each category.
    
    For example, if we have the following `record_id==737` and `record_summary==Counter({3: 1, 2: 1, 1: 1})`, it means that the image with ID `123` has 1 or more objects of category `1`, 1 or more objects of category `2` and 1 or more objects of category `3` respectively. This should help you to pick an image including a specific set of categories.

    ### Exploring Individual Record

    The individual record exploration produces the following output.

    ```python
    from squids import explore_tfrecords

    record_image, record_summary = explore_tfrecords(
        "dataset/synthetic-tfrecords/instances_train",
        image_id=737,
        with_categories=True,
        with_bboxes=True,
        with_segmentations=True,
        return_artifacts=True
    )

    print("record_summary:", record_summary)
    record_image
    ```

    ```text
    record_summary: {'image_id': 737, 'image_size': (256, 256), 'number_of_objects': 3, 'available_categories': {1, 2, 3}}
    ``

    ![image with overlays](images/image_with_overlays.png)

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
        If the `IMAGE_ID` is:
        - defined - this command outputs the specified record summary and stores the reconstructed image (with overlays for categories, bounding boxes, and masks) to the `OUTPUT_DIR`;
        - not defined - this command lists summaries for all TFRecords.

    ## Outcome

    ### Exploring All Records

    The result of using this command for exploring all TFRecords is shown below.
     
    ```text
    dataset/synthetic-tfrecords/instances_train/
    725 {1}        921 {1}        485 {1, 3}     686 {1, 2}     166 {1}      
    726 {2}        922 {1, 2, 3}  486 {3}        687 {1}        167 {3}      
    727 {3}        923 {2}        488 {1, 2}     688 {1, 3}     168 {1}      
    728 {2, 3}     924 {2}        489 {2, 3}     689 {1, 2}     169 {2}      
    729 {1, 2}     925 {3}        490 {2, 3}     690 {1, 3}     172 {1}      
    730 {3}        926 {2, 3}     491 {3}        692 {2}        173 {1, 3}   
    731 {1, 2}     927 {1}        493 {1, 2}     693 {3}        174 {1, 3}   
    733 {2}        929 {1, 3}     494 {3}        694 {2}        175 {2}      
    734 {2, 3}     930 {1}        495 {1, 3}     698 {2}        178 {1, 3}   
    735 {1, 2}     933 {1}        496 {1, 3}     699 {1, 2, 3}  179 {1, 2, 3}
    737 {1, 2, 3}  934 {1, 2, 3}  498 {1}        700 {2, 3}     180 {2}      
    739 {2}        935 {3}        499 {1, 3}     702 {2, 3}     181 {3}      
    740 {2}        936 {1, 3}     500 {3}        703 {3}        182 {1, 2, 3}
    741 {1, 2}     937 {1, 2, 3}  501 {1, 2, 3}  706 {1}        183 {2}      
    743 {2}        939 {1, 2}     502 {1}        707 {1, 2}     184 {3}      
    744 {2}        941 {2}        503 {2}        709 {2, 3}     185 {1, 2}   
    745 {1}        943 {2, 3}     504 {1, 3}     711 {2, 3}     186 {1, 2, 3}
    746 {2, 3}     945 {1, 3}     506 {2}        712 {2}        187 {2, 3}   
    747 {1, 2}     948 {1}        507 {1, 2, 3}  713 {3}        188 {2}      
    748 {1, 3}     949 {1, 2}     511 {1}        715 {1, 2}     189 {1, 2}   
    749 {1, 2}     950 {1, 3}     512 {2}        717 {2, 3}     190 {2, 3}   
    750 {1, 2}     952 {1, 3}     514 {1, 2}     718 {2, 3}     192 {1, 3}   
    751 {2, 3}     953 {1, 3}     515 {1, 2}     719 {3}        193 {1, 2}   
    753 {1}        954 {2, 3}     517 {1, 3}     720 {2}        194 {1}      
    ...            ...            ...            ...            ...
    908 {1, 3}     470 {3}        667 {3}        155 {3}        359 {3}      
    910 {2, 3}     472 {2}        668 {1, 2}     156 {1, 2}     361 {2, 3}   
    911 {1, 2}     473 {2, 3}     670 {2, 3}     158 {1}        362 {1, 2, 3}
    912 {1, 2}     477 {2}        671 {1, 2, 3}  159 {3}        363 {1}      
    913 {3}        478 {2}        673 {1, 2}     160 {1, 2, 3}  365 {3}      
    914 {2, 3}     480 {2}        678 {1, 2}     161 {2, 3}     366 {1}      
    915 {1, 3}     481 {1}        680 {2}        162 {1, 2, 3}  367 {1}      
    917 {1, 2}     482 {2}        682 {1}        163 {1, 3}   
    918 {1, 2}     483 {2, 3}     683 {2, 3}     164 {2, 3}   
    920 {1, 2, 3}  484 {3}        684 {1}        165 {1, 2}   
    Total 712 records
    ```

    From the output, you can observe the total number of images contained with these records, the listing of all images IDs combined with indicators categories present for each image. For example, if we have the following line `737 {1,2,3}`, it means that the image with ID `123` has 1 or more objects of category `1`, 1 or more objects of category `2` and 1 or more objects of category `3` respectively. This should help you to pick an image including a specific set of categories.

    ### Exploring Individual Record

    The individual record exploration produces the following output to the console and saves the image file `<image_id>.png` to the `OUTPUT_DIR` directory.

    ```text
    Property              Value
    --------------------  ----------
    image_id              737
    image_size            (256, 256)
    number_of_objects     3
    available_categories  {1, 2, 3}
    Image saved to ./737.png
    ```

    The output contains information about the image identifier its shape, number of annotated objects, and their categories. Also, all information about bounding boxes, segmentation, and categories are overlaid to the image, which example is shown below.

    ![image with overlays](images/image_with_overlays.png)

## PyDoc

::: squids.tfrecords.explorer
    selection:
      members:
        - explore_tfrecords
