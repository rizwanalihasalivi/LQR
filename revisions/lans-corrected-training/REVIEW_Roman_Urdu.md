# LANS review ka nateeja

Code ke concrete training/evaluation issues fix hue hain, lekin **har scenario
mein first aane ka target is experiment mein achieve nahin hua**.

## Kya fix kiya

- PPO 100352 steps chal rahi thi lekin curriculum budget 200704 tha. Is wajah se
  high-delay training stage miss ho rahi thi. Actual budget aur delay coverage
  ab record hote hain; fine-tuning shuru se complete delay distribution use karti hai.
- Output-layer constraint pehle rollout se pehle lagta tha. Ab har optimizer
  update ke baad projection hoti hai.
- Preview mein float32 rounding ke baad subtraction se precision lose ho rahi
  thi. Ab calculation float64 mein hoti hai aur end par float32 conversion hoti hai.
- Wrong controller name, missing MPC comparison, selected PPO reporting aur
  result-overwrite safeguards bhi fix hue hain.

Framework diagram ke blocks, 11 inputs, 64x64 policy aur LQR + PPO residual +
latency gain + saturation structure preserve hain. Koi deployed predictor,
extra memory, queue input ya scenario-wise controller switch add nahin kiya.

## Fresh tests

Chhe naye training runs ne har ek mein 100352 PPO steps complete kiye. Har seed
ne 200 ms delay dekha. Final testing se pehle 18 new/previous checkpoints ke
hashes freeze hue. Sab controllers ko identical physical cases diye gaye.
**2940 final episodes aur 2940 raw traces** verify hue. Training, evaluation aur
precision ke **19 regression tests** pass hue.

| Policy | Complete episodes | Mean penalized RMSE |
|---|---:|---:|
| Previous LANS | 420/420 | 0.026412 m |
| Naya experimental LANS | 418/420 | 0.044852 m |
| Equally pretrained naya PPO | 409/420 | 0.078496 m |

Naya LANS objective-matched four-controller pool mein 5/14 conditions mein first
hai. Sab retained references, previous policies aur BC ablations ko aik saath
include karne par naya candidate 0/14 mein first hai. Kisi strong baseline ko
hata kar target complete claim nahin kiya gaya.

Naye gains ne no-delay tracking improve ki, lekin mixed/high-delay robustness
regress hui. Dono naye failures mixed-delay slalom ke same physical case mein
policy seeds 83 aur 84 par hue. Purani negative steering FIFO queue se apply ho
rahi thi jab current command positive thi. Heading error 0.5 rad limit cross hui;
signals finite rahe. Policy seed 82 same case complete karta hai. Yeh policy
robustness/generalization failure hai; trace audit mein simulator defect identify
nahin hua.

## Recommendation

**Previous stable LANS policies retain karein.** Naye candidate ko default
replacement na samjhein. Bug fixes valid hain; aggressive retraining ki superiority
prove nahin hui. Model-informed pretraining ka faida aur PPO ka alag learning
faida aik hi cheez nahin hain; BC-only comparisons bhi report mein maujood hain.

MPC ke paas 20 future reference samples hain, policy ke paas 4. Teacher nominal
1650 kg model use karta hai jabke mass policy ko directly observe nahin hota.
Is liye first-place ka masla sirf aik syntax error ya gain barhane se solve nahin
hota. Current results har scenario mein superiority ki guarantee support nahin
karte. Final tests dekh kar parameters dobara tune nahin kiye gaye.

Full [measured report](REVIEW.md), [protocol](results/protocol.json),
[all-reference rankings](results/test/rank_all_references.csv) aur
[integrity audit](results/test/audit.json) saath retain hain.
