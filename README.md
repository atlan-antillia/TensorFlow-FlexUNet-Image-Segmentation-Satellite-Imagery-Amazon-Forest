<h2>TensorFlow-FlexUNet-Image-Segmentation-Satellite-Imagery-Amazon-Forest  (2026/01/01)</h2>
Toshiyuki Arai<br>
Software Laboratory antillia.com<br><br>
This is the first experiment of Image Segmentation for <b></b> based on our <a href="./src/TensorFlowFlexUNet.py">TensorFlowFlexUNet</a> 
(TensorFlow Flexible UNet Image Segmentation Model for Multiclass) , 
and a 512x512 pixels <a href="https://drive.google.com/file/d/1vuZGRdvQGBL8D-55MATqLVJHu17PbFk7/view?usp=sharing">Augmented-Amazon-Forest-ImageMask-Dataset.zip</a> which was derived by us from
<br><br>
<a href="https://zenodo.org/records/4498086/files/AMAZON.rar?download=1">AMAZON.rar</a> in 
<a href="https://zenodo.org/records/4498086">Amazon and Atlantic Forest image datasets for semantic segmentation </a> on zenodo.org.
<br><br>

<b>Data Augmentation Strategy</b><br>
To address the limited size of images and masks of the original <b>Amazon-Forest</b> dataset, which contains 699 GeoTIFF images and TIF masks  respectively,
we generated  the  Augmented dataset by using our <a href="https://github.com/sarah-antillia/Barrel-Image-Distortion-Tool">
Barrel-Image-Distortion-Tool</a>.
<br><br> 

<hr>
<b>Actual Image Segmentation for Amazon-Forest Images of 512x512 pixels</b><br>
As shown below, the inferred masks predicted by our segmentation model trained by the dataset appear similar to the ground truth masks, but they lack precision in certain areas.
<br><br>
<table>
<tr>
<th>Input: image</th>
<th>Mask (ground_truth)</th>
<th>Prediction: inferred_mask</th>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Amazon-Forest/mini_test/images/10388.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Amazon-Forest/mini_test/masks/10388.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Amazon-Forest/mini_test_output/10388.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/Amazon-Forest/mini_test/images/10412.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Amazon-Forest/mini_test/masks/10412.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Amazon-Forest/mini_test_output/10412.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/Amazon-Forest/mini_test/images/10559.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Amazon-Forest/mini_test/masks/10559.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Amazon-Forest/mini_test_output/10559.png" width="320" height="auto"></td>
</tr>
</table>
<hr>
<br>
<h3>1  Dataset Citation</h3>
The dataset used here was derived from <br><br>
<a href="https://zenodo.org/records/4498086/files/AMAZON.rar?download=1">AMAZON.rar</a> in 
<a href="https://zenodo.org/records/4498086">Amazon and Atlantic Forest image datasets for semantic segmentation </a> on zenodo.org.
<br><br>
Bragagnolo, Lucimara, da Silva, Roberto Valmir, Grzybowski, José Mario Vicensi<br>
<br>
This database contains images from <b>Amazon</b> and <b>Atlantic Forest</b> brazilian
 biomes used for training a fully convolutional neural network for the semantic segmentation of forested areas in images from the 
 <b>Sentinel-2 Level 2A Satellite</b>.
 <br>
 The images refer to the composition of bands 4, 3, 2 and 8. Each band was converted to a byte type (0-255).<br><br>
 The images are still divided into three main sets: training, validation and testing:<br>
<ul>
<li>1. <b>Training dataset:</b> it contains 499 and 485 GeoTIFF images (Amazon and Atlantic Forest, respectively) with 512x512 pixels and associated PNG masks (forest indicated in white and background in black color).
</i>
<li>2. <b>Validation dataset:</b> it contains 100 GeoTIFF images for each biome with 512x512 pixels and associated PNG masks used for validation step.
</li>
<li>3. <b>Test dataset:</b> it contains 20 GeoTIFF images for each biome with 512x512 pixels for testing.
</li>
</ul>

<b>Citation</b><br>
Bragagnolo, L., da Silva, R. V., & Grzybowski, J. M. V. (2021). <br>
Amazon and Atlantic Forest image datasets for semantic segmentation [Data set]. <br>
Zenodo. https://doi.org/10.5281/zenodo.4498086
<br>
<br>
<b>License</b><br>
<a href="https://creativecommons.org/licenses/by/4.0/legalcode">Creative Commons Attribution 4.0 International</a>
<br>
<br>
<h3>
2 Amazon-Forest ImageMask Dataset
</h3>
 If you would like to train this Amazon-Forest Segmentation model by yourself,
please down load our 
<a href="https://drive.google.com/file/d/1vuZGRdvQGBL8D-55MATqLVJHu17PbFk7/view?usp=sharing">Augmented-Amazon-Forest-ImageMask-Dataset.zip</a> on the google drive,
expand the downloaded, and put it under <b>./dataser</b> folder to be.<br>
<pre>
./dataset
└─Amazon-Forest
    ├─test
    │   ├─images
    │   └─masks
    ├─train
    │   ├─images
    │   └─masks
    └─valid
        ├─images
        └─masks
</pre>
<br>
<b>Amazon-Forest Statistics</b><br>
<img src ="./projects/TensorFlowFlexUNet/Amazon-Forest/Amazon-Forest_Statistics.png" width="512" height="auto"><br>
<br>
As shown above, the number of images of train and valid datasets is large enough to use for a training set of our segmentation model.
<br><br>
<b>Train_images_sample</b><br>
<img src="./projects/TensorFlowFlexUNet/Amazon-Forest/asset/train_images_sample.png" width="1024" height="auto">
<br>
<b>Train_masks_sample</b><br>
<img src="./projects/TensorFlowFlexUNet/Amazon-Forest/asset/train_masks_sample.png" width="1024" height="auto">
<br>
<h3>
3 Train TensorflowFlexUNet Model
</h3>
 We trained Amazon-Forest TensorflowFlexUNet Model by using the following
<a href="./projects/TensorFlowFlexUNet/Amazon-Forest/train_eval_infer.config"> <b>train_eval_infer.config</b></a> file. <br>
Please move to ./projects/TensorFlowFlexUNet/Amazon-Forest and run the following bat file.<br>
<pre>
>1.train.bat
</pre>
, which simply runs the following command.<br>
<pre>
>python ../../../src/TensorFlowFlexUNetTrainer.py ./train_eval_infer.config
</pre>
<hr>

<b>Model parameters</b><br>
Defined a small <b>base_filters=16</b> and a large <b>base_kernels=(11,11)</b> for the first Conv Layer of Encoder Block of 
<a href="./src/TensorflowUNet.py">TensorflowUNet.py</a> 
and a large num_layers (including a bridge between Encoder and Decoder Blocks).
<pre>
[model]
image_width    = 512
image_height   = 512
image_channels = 3
input_normalize = True
normalization  = False

num_classes    = 2

base_filters   = 16
base_kernels  = (11,11)
num_layers    = 8

dropout_rate   = 0.05
dilation       = (1,1)
</pre>

<b>Learning rate</b><br>
Defined a small learning rate.  
<pre>
[model]
learning_rate  = 0.00007
</pre>

<b>Loss and metrics functions</b><br>
Specified "categorical_crossentropy" and "dice_coef_multiclass".<br>
<pre>
[model]
loss           = "categorical_crossentropy"
metrics        = ["dice_coef_multiclass"]
</pre>
<b >Learning rate reducer callback</b><br>
Enabled learing_rate_reducer callback, and a small reducer_patience.
<pre> 
[train]
learning_rate_reducer = True
reducer_factor     = 0.4
reducer_patience   = 4
</pre>
<b>Early stopping callback</b><br>
Enabled early stopping callback with patience parameter.
<pre>
[train]
patience      = 10
</pre>
<b></b><br>
<b>RGB color map</b><br>
rgb color map dict for Amazon-Forest 1+1 classes.<br>
<pre>
[mask]
mask_file_format = ".png"
;Amazon-Forest 1+1
rgb_map = {(0,0,0):0,  (255, 255, 255):1, }       
</pre>
<b>Epoch change inference callbacks</b><br>
Enabled epoch_change_infer callback.<br>
<pre>
[train]
epoch_change_infer       = True
epoch_change_infer_dir   =  "./epoch_change_infer"
epoch_changeinfer        = False
epoch_changeinfer_dir    = "./epoch_changeinfer"
num_infer_images         = 6
</pre>
By using this epoch_change_infer callback, on every epoch_change, the inference procedure can be called
 for 6 images in <b>mini_test</b> folder. This will help you confirm how the predicted mask changes 
 at each epoch during your training process.<br> <br> 
<b>Epoch_change_inference output at starting (1,2,3)</b><br>
<img src="./projects/TensorFlowFlexUNet/Amazon-Forest/asset/epoch_change_infer_at_start.png" width="1024" height="auto"><br>
<br>
<b>Epoch_change_inference output at ending (38,39,40)</b><br>
<img src="./projects/TensorFlowFlexUNet/Amazon-Forest/asset/epoch_change_infer_at_middlepoint.png" width="1024" height="auto"><br>
<br>
<b>Epoch_change_inference output at ending (78,79,80)</b><br>
<img src="./projects/TensorFlowFlexUNet/Amazon-Forest/asset/epoch_change_infer_at_end.png" width="1024" height="auto"><br>

<br>
In this experiment, the training process was terminated at epoch 80.<br><br>
<img src="./projects/TensorFlowFlexUNet/Amazon-Forest/asset/train_console_output_at_epoch80.png" width="880" height="auto"><br>
<br>
<a href="./projects/TensorFlowFlexUNet/Amazon-Forest/eval/train_metrics.csv">train_metrics.csv</a><br>
<img src="./projects/TensorFlowFlexUNet/Amazon-Forest/eval/train_metrics.png" width="520" height="auto"><br>

<br>
<a href="./projects/TensorFlowFlexUNet/Amazon-Forest/eval/train_losses.csv">train_losses.csv</a><br>
<img src="./projects/TensorFlowFlexUNet/Amazon-Forest/eval/train_losses.png" width="520" height="auto"><br>
<br>
<h3>
4 Evaluation
</h3>
Please move to a <b>./projects/TensorFlowFlexUNet/Amazon-Forest</b> folder,<br>
and run the following bat file to evaluate TensorflowFlexUNet model for Amazon-Forest.<br>
<pre>
>./2.evaluate.bat
</pre>
This bat file simply runs the following command.
<pre>
>python ../../../src/TensorFlowFlexUNetEvaluator.py  ./train_eval_infer.config
</pre>
Evaluation console output:<br>
<img src="./projects/TensorFlowFlexUNet/Amazon-Forest/asset/evaluate_console_output_at_epoch80.png" width="880" height="auto">
<br><br>Image-Segmentation-Amazon-Forest

<a href="./projects/TensorFlowFlexUNet/Amazon-Forest/evaluation.csv">evaluation.csv</a><br>
The loss (categorical_crossentropy) to this Amazon-Forest/test was not low, but dice_coef_multiclass  high as shown below.
<br>
<pre>
categorical_crossentropy,0.0358
dice_coef_multiclass,0.9785
</pre>
<br>
<h3>5 Inference</h3>
Please move to a <b>./projects/TensorFlowFlexUNet/Amazon-Forest</b> folder<br>
,and run the following bat file to infer segmentation regions for images by the Trained-TensorflowFlexUNet model for Amazon-Forest.<br>
<pre>
>./3.infer.bat
</pre>
This simply runs the following command.
<pre>
>python ../../../src/TensorFlowFlexUNetInferencer.py ./train_eval_infer.config
</pre>
<hr>
<b>mini_test_images</b><br>
<img src="./projects/TensorFlowFlexUNet/Amazon-Forest/asset/mini_test_images.png" width="1024" height="auto"><br>
<b>mini_test_mask(ground_truth)</b><br>
<img src="./projects/TensorFlowFlexUNet/Amazon-Forest/asset/mini_test_masks.png" width="1024" height="auto"><br>
<hr>
<b>Inferred test masks</b><br>
<img src="./projects/TensorFlowFlexUNet/Amazon-Forest/asset/mini_test_output.png" width="1024" height="auto"><br>
<br>
<hr>
<b>Enlarged images and masks for Amazon-Forest Images of 512x512 pixels</b><br>
As shown below, the inferred masks predicted by our segmentation model trained by the dataset appear similar to the ground truth masks, but they lack precision in certain areas.
<br>
<br>
<table>
<tr>
<th>Input: image</th>
<th>Mask (ground_truth)</th>
<th>Prediction: inferred_mask</th>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Amazon-Forest/mini_test/images/10255.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Amazon-Forest/mini_test/masks/10255.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Amazon-Forest/mini_test_output/10255.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/Amazon-Forest/mini_test/images/10338.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Amazon-Forest/mini_test/masks/10338.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Amazon-Forest/mini_test_output/10338.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/Amazon-Forest/mini_test/images/10548.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Amazon-Forest/mini_test/masks/10548.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Amazon-Forest/mini_test_output/10548.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Amazon-Forest/mini_test/images/10498.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Amazon-Forest/mini_test/masks/10498.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Amazon-Forest/mini_test_output/10498.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Amazon-Forest/mini_test/images/barrdistorted_1001_0.3_0.3_10125.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Amazon-Forest/mini_test/masks/barrdistorted_1001_0.3_0.3_10125.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Amazon-Forest/mini_test_output/barrdistorted_1001_0.3_0.3_10125.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Amazon-Forest/mini_test/images/barrdistorted_1001_0.3_0.3_10326.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Amazon-Forest/mini_test/masks/barrdistorted_1001_0.3_0.3_10326.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Amazon-Forest/mini_test_output/barrdistorted_1001_0.3_0.3_10326.png" width="320" height="auto"></td>
</tr>
</table>
<hr>
<br>
<h3>
References
</h3>

<b>1.  An attention-based U-Net for detecting deforestation within satellite 
sensor imagery</b><br>
David John, Ce Zhang<br>
<a href="https://www.sciencedirect.com/science/article/pii/S0303243422000113">https://www.sciencedirect.com/science/article/pii/S0303243422000113</a>
<br>
<br>
<b>2.  FOREST SEMANTIC SEGMENTATION BASED ON DEEP LEARNING USING 
SENTINEL-2 IMAGES </b><br>
C. Hızal,  G. Gülsu, H. Y. Akgün,  B. Kulavuz, T. Bakırman, A. Aydın, B. Bayram<br>
<a href="https://isprs-archives.copernicus.org/articles/XLVIII-4-W9-2024/229/2024/isprs-archives-XLVIII-4-W9-2024-229-2024.pdf">
https://isprs-archives.copernicus.org/articles/XLVIII-4-W9-2024/229/2024/isprs-archives-XLVIII-4-W9-2024-229-2024.pdf</a> 
<br><br>

<b>3. TensorFlow-FlexUNet-Image-Segmentation-Model</b><br>
Toshiyuki Arai <br>
<a href="https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-Model">
TensorFlow-FlexUNet-Image-Segmentation-Model
</a>
<br>
<br>
