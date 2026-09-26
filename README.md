# image_picker

dataset is created where  [laplacian_var, brightness_mean, clip_shadow_pct, clip_highlight_pct, histogram_std, has_face, blendshape_vector(52), saliency_x, saliency_y, thirds_dist, CLS_token(768)]  this to total   stored in features.npz
but issue is that svm or xgboost cclassifer is not working properly  



now Quick test of NIMA (trained on AVA, ~255k images) and MUSIQ aesthetic
scoring on a handful of your photos, via the `pyiqa` toolbox -- no need to
hunt for weights or convert Keras checkpoints by hand  stored in img_vit_sataset.npz

this is qwen vl 2b parameter model now which sotred the vectors in dataset_vlm.npz 
==================================================
ZERO-SHOT VLM ROC-AUC: 0.6267
Optimal Threshold: 0.8359
==================================================
              precision    recall  f1-score   support

REJECTED (0)       0.86      0.58      0.69       233
SELECTED (1)       0.34      0.69      0.45        72

    accuracy                           0.60       305
   macro avg       0.60      0.63      0.57       305
weighted avg       0.74      0.60      0.63       305
in all three the accuracy is not good 
 