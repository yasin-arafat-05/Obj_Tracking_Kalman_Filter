<br>

---

<br>

# **Kalman Filter এর বেসিক ধারণা:**

<br>

---

<br>

Kalman Filter একটি এলগোরিদম যা ভবিষ্যতের অবস্থার পূর্বাভাস দেয় এবং সেন্সর থেকে প্রাপ্ত তথ্য ব্যবহার করে পূর্বাভাসকে আরো নির্ভুল করে। এটি দুইটি ধাপে কাজ করে:

1. **Prediction (পূর্বাভাস)**:  
   - ভবিষ্যতের অবস্থার অনুমান করে।  
   - এটি সিস্টেমের গতিশীলতার (motion dynamics) উপর নির্ভরশীল।  

2. **Correction (সংশোধন)**:  
   - সেন্সর থেকে প্রাপ্ত মানের সাথে পূর্বাভাসের তুলনা করে।  
   - ত্রুটি (error) কমিয়ে আপডেটেড অবস্থান নির্ণয় করে।

---

### **Kalman Filter-এর উপাদান**

1. **State Vector (অবস্থার ভেক্টর)**:  
   এটি এমন একটি ভেক্টর যা সিস্টেমের বর্তমান অবস্থাকে বোঝায়। উদাহরণস্বরূপ, চলমান বস্তুর জন্য এটি হতে পারে:  

   $\text{State Vector:} \begin{bmatrix} x \\ y \\ v_x \\ v_y \end{bmatrix}$

   যেখানে x, y পজিশন এবং $v_x$, $v_y$ গতির উপাদান।

2. **Transition Matrix (F)**:  
   সিস্টেমের বর্তমান অবস্থা থেকে ভবিষ্যতের অবস্থায় কীভাবে যাবে তা নির্ধারণ করে।  

3. **Measurement Matrix (H)**:  
   সেন্সর কী কী পরিমাপ করতে পারে তা নির্ধারণ করে। উদাহরণস্বরূপ, সেন্সর কেবল পজিশন মাপতে পারে।  

4. **Process Noise (Q)**:  
   সিস্টেমের পূর্বাভাসের ত্রুটি।  

5. **Measurement Noise (R)**:  
   সেন্সরের ত্রুটি।  

<br>

---

ধরি, আপনি **ড্রোন ট্র্যাকিং** এর একটি প্রজেক্ট করছেন। ড্রোন আকাশে উড়ছে এবং তার অবস্থান (পজিশন) ও গতি (ভেলোসিটি) নির্ধারণ করতে Kalman Filter ব্যবহার করবেন।  

---

### **বাস্তব উদাহরণ: ড্রোন ট্র্যাকিং**

#### ১. **State Vector (অবস্থার ভেক্টর)**  
ড্রোনের অবস্থান ও গতি:  

$\text{State Vector:} \begin{bmatrix} x \\ y \\ v_x \\ v_y \end{bmatrix}$

- x, y: ড্রোনের বর্তমান অবস্থান।  
- $v_x$, $v_y$ : ড্রোনের চলার গতি (X এবং Y অক্ষ বরাবর)।  

**বাস্তব জীবনে:**  
আপনার ড্রোন GPS ব্যবহার করে প্রতি সেকেন্ডে তার x, y অবস্থান জানাচ্ছে। কিন্তু GPS ধীর গতির এবং কখনো কখনো সঠিক তথ্য দেয় না। Kalman Filter GPS-এর তথ্য ব্যবহার করে সঠিক গতিপথ অনুমান করতে পারে।  

---

#### ২. **Transition Matrix (F)**  
এটি ড্রোনের গতিশীলতার মডেল:  

F = $\begin{bmatrix} 
1 & 0 & 1 & 0 \\ 
0 & 1 & 0 & 1 \\ 
0 & 0 & 1 & 0 \\ 
0 & 0 & 0 & 1 
\end{bmatrix}$
  
- $x_{new} = x_{old} + v_x \times dt$ 
- $y_{new} = y_{old} + v_y \times dt$

**বাস্তব জীবনে:**  
ড্রোন যদি প্রতি ১ সেকেন্ডে $v_x$ = 5, $\text{m/s}$ এবং $v_y$ = 2 , $\text{m/s}$ গতিতে চলে, তাহলে তার পরবর্তী অবস্থান হবে:  

$x_{new}$ = x + 5 , $\text{(m)}$  


$y_{new}$ = y + 2, $\text{(m)}$
  

---

#### ৩. **Measurement Matrix (H)**  

H = $\begin{bmatrix} 
1 & 0 & 0 & 0 \\ 
0 & 1 & 0 & 0 
\end{bmatrix}$
  
এটি শুধুমাত্র x, y পজিশনকে মাপতে ব্যবহার করে।  

**বাস্তব জীবনে:**  
GPS শুধুমাত্র x, y অবস্থান দেয়, কিন্তু Kalman Filter $v_x$, $v_y$ গতিও অনুমান করে।  

---

#### ৪. **Process Noise (Q)**  
ড্রোনের গতি বা অবস্থানের ত্রুটি।  

**বাস্তব জীবনে:**  
ড্রোন বাতাসের কারণে একটু কেঁপে উঠতে পারে বা গতি হঠাৎ পরিবর্তন হতে পারে। এই ত্রুটিকে Q দিয়ে মডেল করা হয়।  

---

#### ৫. **Measurement Noise (R)**  
GPS বা সেন্সরের ত্রুটি।  

**বাস্তব জীবনে:**  
GPS কখনো কখনো সঠিক পজিশন দিতে ব্যর্থ হয়, যেমন কোনো বড় বিল্ডিং বা গাছের নিচে। এই ত্রুটি R ম্যাট্রিক্সে নির্ধারণ করা হয়।  

---

### **Kalman Filter বাস্তবায়ন: Python কোড (ড্রোন ট্র্যাকিং)**

```python
import cv2
import numpy as np

# Kalman Filter সেটআপ
kf = cv2.KalmanFilter(4, 2)  # 4 State Variables: x, y, vx, vy | 2 Measurements: x, y

# State Transition Matrix (F)
kf.transitionMatrix = np.array([[1, 0, 1, 0],
                                 [0, 1, 0, 1],
                                 [0, 0, 1, 0],
                                 [0, 0, 0, 1]], np.float32)

# Measurement Matrix (H)
kf.measurementMatrix = np.array([[1, 0, 0, 0],
                                  [0, 1, 0, 0]], np.float32)

# Process Noise (Q)
kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03

# Measurement Noise (R)
kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.5

# Initial State
kf.statePost = np.array([[0], [0], [0], [0]], np.float32)

# ড্রোনের GPS ডেটা (উদাহরণস্বরূপ)
gps_data = [(100, 200), (105, 205), (110, 210), (116, 215), (120, 220)]

for i, measurement in enumerate(gps_data):
    # পূর্বাভাস (Prediction)
    predicted = kf.predict()

    # সংশোধন (Correction)
    measurement = np.array([[measurement[0]], [measurement[1]]], np.float32)
    updated = kf.correct(measurement)

    print(f"Frame {i+1}:")
    print(f"Predicted Position: {predicted[:2].flatten()}")
    print(f"Measured Position: {measurement.flatten()}")
    print(f"Updated Position: {updated[:2].flatten()}")
    print("-" * 50)
```

---

### **বাস্তব প্রয়োগ**
- **ড্রোন ট্র্যাকিং**: Kalman Filter GPS ডেটা থেকে ড্রোনের মসৃণ অবস্থান ও গতিপথ অনুমান করতে ব্যবহার করা হয়।  
- **Robot Navigation**: রোবটের অবস্থান নির্ধারণ এবং গতিপথ নিয়ন্ত্রণ।  
- **Autonomous Vehicles**: স্বয়ংক্রিয় গাড়ির সেন্সর ফিউশন (GPS, Lidar, Radar)।  
- **Finance**: শেয়ারের দাম পূর্বাভাস।  

---
