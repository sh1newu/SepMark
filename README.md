# SepMark
The implementation of the paper "SepMark: Deep Separable Watermarking for Unified Source Tracing and Deepfake Detection".  
Paper Link: https://doi.org/10.1145/3581783.3612471

**Quick Test (2/23/2024)**
Due to the inconvenience of deploying Deepfake models, you may want to quickly test the robustness under common or custom distortions. In such cases, it is suggested that you comment out the lines regarding deepfake distortions in the file network/noise_layers/__init__.py. Meanwhile, in the file test_Dual_Mark.py, you should modify noise_layers_F = args.noise_layers.pool_F to noise_layers_F = []. There might be some negligible performance changes brought by these modifications compared to our resultant tensorboard files in the runs\Tables_2_3_and_6\ folder.

**Update (11/3/2023)**
We have put all the [noise layers](https://drive.google.com/drive/folders/17B02FgS8hYtW3V1GVZkiy0wrq--FeYVf?usp=sharing) and [datasets](https://drive.google.com/drive/folders/1LqvsnoiyyYyrYSmnTRz-y6LP5UheYCXH?usp=sharing) into Google Drive due to the file sizes; see the README file please. Hopefully, all goes well, but we never guarantee that you'll be able to run it directly without any Debug. How to replicate/use the code? For me, when everything above is ready, it is enough to modify the configuration file in the cfg folder and run the test/main file successfully. So, Debug is all you need! If there are any missing uploads, please feel free to contact me.

**Reminder (9/20/2023)**
We provide the models of SepMark that were trained using SimSwap(), GANimation(), and StarGAN (Male) [***here***](https://drive.google.com/drive/folders/1h93NcAJXE21CsDluMyDdBdKGY5aV1pLC?usp=sharing). Due to license agreements, we are unable to distribute their codes ourselves. However, we believe that you can re-implement them or other Deepfakes on your own. Here are some links to their repositories, including [SimSwap](https://github.com/neuralchen/SimSwap), [GANimation](https://github.com/vipermu/ganimation), [StarGAN](https://github.com/yunjey/stargan), [FaceSwap](https://github.com/guipleite/CV2-Face-Swap), [Roop](https://github.com/s0md3v/roop), and [MobileFaceSwap](https://github.com/Seanseattle/MobileFaceSwap). Best Wishes.  

The code is strictly for non-commercial academic use only.  
Contact: xinliao@hnu.edu.cn / shinewu@hnu.edu.cn

If you find our work useful, please consider citing:  

```
@inproceedings{wu2023sepmark,  
  title={SepMark: Deep Separable Watermarking for Unified Source Tracing and Deepfake Detection},  
  author={Wu, Xiaoshuai and Liao, Xin and Ou, Bo},  
  booktitle={Proceedings of the 31th ACM International Conference on Multimedia},  
  year={2023}  
}
```
