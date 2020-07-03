# TTE2020

This project aligns 2D Dicom cardiac MR images into 3D image volumes. For now, it works on data from the [Data Science Bowl Cardiac Challenge Data](https://www.kaggle.com/c/second-annual-data-science-bowl). 

## Dependencies 

* SimpleITK
* Matplotlib

## Example Usage 

```
python align_2d_dicoms.py --folder second-annual-data-science-bowl/train/train  --out_folder second-annual-data-science-bowl/train/aligned --num 5
```

