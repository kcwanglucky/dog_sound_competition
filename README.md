# T-Brain Furbo Dog Sound Classification Competition

### Final Score
    53th/301 
    Get 0.960740 final auc score, on private dataset. (First place: 0.989945)

### Different Trials (w/ Public Score)
    1. Plain approach: Take the melspectrogram of the input audio and then feed it to CNN models. Have tune the parameters such that the highest possible public auc is aound 0.95, while 0.98 on validation set
            Score: 0.9549
    2. Data Augmentation: Try different ways (time_shift, add_white_noise, time_stretch, spec_aug) to augment the data (online data augmentation). Only a minor improve in the auc score.
            Score: 0.9618 
    3. Two models: Given the above result, label 0, 1, 2, have the worse prediction accuracy (often mix with each other). Hence, I try to first predict 6 classes. If pred falls into label 1, 2, 3, then go over another model only for label 0, 1, 2 3-class classification. The 3-class model, however, does not show a huge improvement. 
            Score: 0.8812
    4. Remove insignificant samples: Remove samples with too many empty values (no sound). Threshold values tried (0.3, 0.35, 0.4, 0.5). No improvement
            Score: Around 0.95
    5. Transformer: Parameters do not really move/learn. Haven't yet successfully tune the model such that its gradient won't vanish. Lack of time
            Score: 0.7315

### TODOs
- [x] 1. Try more data augmentation approaches.
- [x] 2. Try different filter shape (Maybe can try a larger filter to extract the more macro view of the audio)
- [ ] 3. Try different model configs, model architecture (ex: transformer)