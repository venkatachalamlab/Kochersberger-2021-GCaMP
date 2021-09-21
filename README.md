# Kochersberger-2021-GCaMP
Code used for calcium data analysis in "Programmed Cell Death Modifies Neural Circuits and Tunes Intrinsic Behavior".  

### Data Acquisition
For data acquisition, a confocal imaging system capable of performing fast volumeric imaging at 4Hz is used.
For more information about the system visit: [Lambda github](https://github.com/venkatachalamlab/lambda)

### Manual Annotation
To annotate neurons in recordings, a costomized software is used. This software uses a web based user interface, and visualizes datasets to make manual annotations easier. The result is saved as a pandas dataframe in a hdf file. For more information about the software visit: [annotator github](https://github.com/venkatachalamlab/annotator)

### Preprocessing
To prepare datasets for trace extraction, first we use a large blurring filter to roughly center neurons in each volume. Next we crop the image to only keep a smaller area around neruons. The cropping step reduces file size. Next, we use annotator (see above) to label all neurons at all timepoints. We use the coordiantes of neurons to extract traces. For trace extraction, we use a fixed number of pixels in each neurons that have the highest intensity. In case there is another neurons close to the neuron of interes, we exclude pixels that are closer to the neighboring neurons than the neuron of interest.

We use the time series for the red fluorescent protein, to subtract artifacts from the time series of the green fluorescent proteins. We apply a low pass filter to reduce the noise, and finally fit a double exponential curve to the time series to correct for the photobleaching effect. These debleached traces are then saved as numpy arrays and can be used for plotting.

### Plotting
Cleaned up traces from the preprocessing step, and the times of reversal initiations (saved as a txt file for each dataset) are used to make plots.
