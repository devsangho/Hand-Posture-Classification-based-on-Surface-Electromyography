# Hand Posture Classification based on Surface Electromyography

<img width="665" alt="hand-postures" src="https://github.com/devsangho/Hand-Posture-Classification-based-on-Surface-Electromyography/assets/54205862/4ab3cae7-55d7-4b3e-9a06-42ad773ef58b">

This project classify hand posture using sEMG signals.

## 1. Segmentation
- Number of channel Number of class Sampling frequency: 12
- Total experiment time (All classes): 17
- Repetition (Session): 2,000 Hz
- Time per repetition: 15 min (1,800,000 points) 6 sessions per class 3-5s
- Duration per repetition : 3~5 s
- Segment size : 200 ms

## 2. 통계적 수치 계산 (Time domain feature)
- 1개의 신호 구간에서 36개의 통계적 수치를 계산한다.
→ [MAV, VAR, WL] x 12 (channels)

$$MAV = \frac{1}{N}\sum_{i=1}^{N} |x_{i}|$$

$$VAR = \frac{1}{N}\sum_{i=1}^{N} |x_{i}^{2}|$$

$$WL = \sum_{i=1}^{N} |x_{i+1} - x_{i}|$$

## 3. Train / Test
- Repetition 1 ~ 4: train set
- Repetition 5 ~ 6: test set

## 4. Result
- Accuracy: 55%

![Figure_2](https://github.com/devsangho/Hand-Posture-Classification-based-on-Surface-Electromyography/assets/54205862/f08ffb09-58d6-4305-9cc4-90150e2df776)

![Figure_1](https://github.com/devsangho/Hand-Posture-Classification-based-on-Surface-Electromyography/assets/54205862/9f83c1f4-9e93-424a-a3d9-02413434bcf3)
