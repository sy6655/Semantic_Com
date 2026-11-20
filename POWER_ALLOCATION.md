# ImageNet 분류: 학습 가능한 Q, 전력 할당과 Rayleigh fading

ImageNet classification 경로의 설명입니다. Cityscapes도 Q/Rayleigh/DDRM 경로를
지원하며, 입력 해상도와 블록 구성 및 실행 방법은 아래 Cityscapes 항목을 참고하시면 됩니다.

latent를 공간상 8×8, 총 64개 블록으로 나누고, 각 블록을 행으로 펼친
`P_p(z) ∈ R^(64×D)`에 `Q^T`를 왼쪽에서 곱합니다. 채널과 블록 내부의
좌표는 유지되고, 64개 블록 축만 섞입니다. 512×512 이미지에서는
VAE latent가 4×64×64이고 블록 하나는 4×8×8입니다.

`L`의 strict upper triangle 2,016개 값만 학습하며,
`Q = matrix_exp(L - L.T)`로 직교성을 유지합니다. 초기 Q는 항등행렬입니다.
Q 파라미터는 allocator state dict의 `orthogonal.upper`에 함께 저장됩니다.

VAE latent는 `latent_dist.sample(generator=generator) * 0.18215`로 생성합니다.
학습·검증·추론 모두 posterior에서 샘플링하며 VAE 파라미터는 고정합니다.
검증·추론에는 seed가 설정된 generator를 전달하므로 동일한 seed로 재현할 수 있습니다.

채널은 perfect CSI를 사용하는 Rayleigh block fading입니다. 매 배치에서
다음과 같이 이미지마다 복소 계수 하나를 생성합니다.

```python
h = (torch.randn(B, device=device) + 1j * torch.randn(B, device=device)) / (2 ** 0.5)
```

`h.shape == [B]`이며 같은 이미지의 64개 블록에는 동일한 h를 적용합니다.
이미지와 전송이 달라질 때 새로 샘플링합니다. 분포는 `h ~ CN(0,1)`이고
`E[|h|²] = 1`입니다. 송수신이 샘플링된 복소 h를 정확히 알고 있다고 가정합니다.
latent의 실수 값을 블록 안에서 두 개씩 I/Q로 묶어 실제 복소 곱셈을 수행합니다.

## 1. Q + allocator + classifier 공동 학습

```bash
python train_power_allocator.py \
  --exp_name imagenet_q64 \
  --dataset ImageNet --data_path /dataset/ImageNet \
  --image_size 512 --num_groups 64 --model ResNet50 \
  --snr_db -5 0 5 10 15 20
```

기본값으로 orthogonal transform, analog transmission, pilot 0개를 사용합니다.
학습 순서는 `VAE → patchify → Q^T → allocator → Rayleigh fading + complex noise → oracle h,a 보정
→ Q → unpatchify → classifier`입니다. VAE는 고정하고 task cross entropy로
Q, allocator, classifier를 공동 학습합니다. 학습 중 diffusion은 사용하지 않습니다.
allocator는 변환된 블록을 보고 전력을 할당하며,
`sum_b a_b^2 ||U_b||² = N_real * P`의 프레임 에너지 제약을 적용합니다.
`P=1`은 실수 성분 하나당 평균 송신 전력입니다. 512×512 이미지에서는
`N_real=4*64*64=16,384`이므로 전체 송신 에너지는 16,384입니다.
I/Q 두 성분을 묶은 complex channel use 수는 8,192이며, complex symbol당
평균 에너지는 2입니다. `mean_real_component_power`와 `mean_symbol_power`로
두 기준을 구분해 기록합니다. allocated와 uniform 전략 모두 같은 에너지 제약을 사용합니다.
allocator에 지정한 학습률은 Q에도 적용됩니다.
분류 allocator, segmentation allocator, gain estimator의 새 학습은 공통으로
optimizer step마다 LR을 갱신합니다. 기본 `--warmup_epochs 0.5` 동안 최대 LR의
10%에서 100%로 선형 증가한 뒤, 남은 학습 기간에는 cosine으로 최대 LR의 50%까지
감소합니다. 다음 옵션을 세 학습 코드에서 동일하게 사용할 수 있습니다.

```bash
--warmup_epochs 0.5 --warmup_start_factor 0.1 --min_lr_ratio 0.5
```

`--lr 1e-3`이면 warmup 시작 LR은 `1e-4`, 최대 LR은 `1e-3`, 최종 LR은
`5e-4`입니다. Allocator와 task model에 다른 LR을 지정하면 각자의 최대 LR에
같은 비율을 적용합니다. Validation과 AMP가 건너뛴 optimizer update에서는 LR을
진행하지 않습니다. 실제 스케줄은 `config.json`의 `lr_schedule`과 `train.log`에 기록합니다.
총 epoch 기본값은 분류 10, segmentation 100, gain estimator 40입니다.
`--resume`은 저장된 optimizer/scheduler를 복원합니다. 기존 epoch-cosine
체크포인트는 기존 스케줄을 유지하며, 새 옵션으로 재개 스케줄을 덮어쓰지 않습니다.

매 epoch 종료 시 `latest.pth`를 갱신하고, 기본 5 epoch마다
`epoch_005.pth`, `epoch_010.pth`처럼 별도 checkpoint를 보관합니다.
마지막 epoch도 별도 보관합니다. `--save_every 10`으로 간격을 바꿀 수 있고,
`--save_every 0`이면 `latest.pth`만 저장합니다. 각 checkpoint에는 Q를 포함한
allocator, classifier, optimizer, scheduler와 AMP scaler 상태가 포함되며
`--resume`으로 재개할 수 있습니다.
송신기의 CSI는 allocator에 자연로그
`log(gamma) = log(|h|²*P/N0)`를 입력하는 방식으로 반영합니다.
`N0 = 10^(-snr_db/10)`는 fading 적용 전의 설정값으로 유지하므로
h를 샘플링한 뒤 잡음 크기를 다시 정규화하지 않습니다.
따라서 fading 전 실수 성분별 SNR은 `P/N0 = 10^(snr_db/10)`이고,
fading 후 이미지별 프레임 SNR은 `|h|²*P/N0`입니다.

allocator는 Learned Transport v4의 블록별 MLP 구조를 사용합니다.

| 구성 | 기본 구조 |
| --- | --- |
| 입력 | 변환된 실수 블록 `U[B,64,256]`, `log(gamma)[B]` |
| 블록별 공유 MLP | `256 → 128 → 64` |
| Global context | 64개 블록 feature의 평균, 64차원 |
| Block ID | 학습 가능한 8차원 embedding |
| 에너지 feature | `log(mean(U_b²)) = log(||U_b||²/256)` |
| 블록별 head 입력 | local 64 + global 64 + ID 8 + energy 1 + log SNR 1 = 138 |
| 공유 출력 head | `138 → 64 → 1`, 전체 출력 `[B,64]` logits |
| 파라미터 수 | allocator 50,625개, 별도 Q 2,016개 |

활성화는 SiLU를 사용합니다. `--hidden_dim`의 기본값은 64이며, 이 값은
블록 feature와 head 폭을 지정합니다. 512×512 이미지의 기본 블록 차원은 256이고,
다른 이미지 크기에서는 실제 블록의 실수 성분 수를 입력 차원으로 사용합니다.

classifier에는 기본적으로 역변환된 수신 latent만 입력합니다.
`--condition_classifier_on_snr`도 지원하며, 이 경우 Q의 제곱으로 블록별
잡음 분산을 공간 좌표로 전파한 뒤 SNR map을 만듭니다.

Q는 allocator 내부 파라미터이므로 allocator와 함께 DDP로 동기화하며,
classifier도 별도로 DDP를 적용합니다. GPU 2개를 사용하는 예시는 다음과 같습니다.
`--batch`는 GPU 하나당 배치 크기이므로 아래 설정의 전체 배치는 32입니다.

```bash
torchrun --standalone --nproc_per_node=2 train_power_allocator.py \
  --exp_name imagenet_q64 --dataset ImageNet --data_path /dataset/ImageNet \
  --batch 16 --image_size 512 --num_groups 64 --model ResNet50 \
  --snr_db -5 0 5 10 15 20
```

## 2. 변환된 블록별 gain estimator 학습

```bash
python train_gain_estimator.py \
  --allocator_checkpoint checkpoints_power/ImageNet/snr_-5.0-0.0-5.0-10.0-15.0-20.0/imagenet_q64/latest.pth \
  --exp_name imagenet_q64_gain \
  --snr_db_min -5 --snr_db_max 20
```

이 단계에서는 해당 checkpoint의 VAE, Q, allocator를 고정합니다.
입력 전처리에서 perfect CSI로 `R_h = Y/h`를 만들어 h의 위상과 크기를 보정합니다.
이 단계에서는 송신 gain `a_b`를 제거하지 않습니다.
estimator는 `R_h`, `log(nu)`, `log(P)`를 받아 블록별 `log(a_hat_b)`를 출력합니다.
여기서 `nu = N0/|h|²`, `gamma = P/nu`, `P=1`입니다.
목표는 timestep이나 공간 패치의 gain이 아니라 **Q^T 공간의 64개 블록별 gain**이며,
학습 loss는 `MSE(log(a_hat), log(a))`입니다. 수신 입력을 구성할 때
oracle gain이나 clean latent를 estimator에 제공하지 않습니다.

estimator는 각 블록을 독립적인 길이 256의 실수 sequence로 처리한 뒤,
전체 블록 정보를 하나의 MLP에서 결합합니다.

| 구성 | 기본 구조 |
| --- | --- |
| 블록별 공유 Conv1D | `1 → 16 → 32`, 두 계층 모두 kernel 5, padding 2, SiLU |
| Conv1D pooling | 블록 내부 sequence 방향 평균, 32차원 |
| 블록별 통계량 | Section VI-B의 6개 통계량 |
| Block ID | 학습 가능한 8차원 embedding |
| 블록당 feature | 32 + 6 + 8 = 46차원 |
| Global head 입력 | 64 × 46 + log(nu) + log(P) = 2,946차원 |
| 출력 MLP | `2946 → 256 → 64`, 출력은 블록별 log gain |
| 파라미터 수 | 774,080개 |

통계량의 순서는 `mean`, `log(M2)`, `mean(abs(r))`, `log(M4/M2²)`,
`C1/M2`, `log(M2-mean²)`입니다. `Mk=mean(r^k)`이고
`C1=mean(r[j]*r[j+1])`는 인접한 m-1개 곱의 평균입니다.
로그 및 분모에는 `1e-8` 하한을 적용합니다.
Conv1D에는 `R_h`를 직접 입력하며, 알려진 잡음 분산은 `log(nu)`로 제공합니다.
`--hidden_dim`은 global MLP의 hidden 폭이며 기본값은 256입니다.

학습 결과는 `checkpoints_gain_estimator/ImageNet/imagenet_q64_gain/latest.pth`에
저장됩니다. `torchrun`을 통한 분산 학습과 `--resume`을 지원합니다.
Estimator는 학습에 사용한 Q/allocator의 fingerprint와 함께 저장됩니다.
Classification inference와 시각화에서는 다른 송신기 checkpoint도 사용하실 수
있으며, 실제 사용한 fingerprint와 학습 당시의 일치 여부를 결과 JSON에 기록합니다.
Estimator 학습 재개 시에는 기존처럼 같은 송신기인지 검사합니다.

gain estimator도 DDP로 학습할 수 있습니다. 이 단계에서 고정된 Q/allocator와
VAE는 각 GPU에서 추론만 수행합니다.

```bash
torchrun --standalone --nproc_per_node=2 train_gain_estimator.py \
  --allocator_checkpoint checkpoints_power/ImageNet/snr_-5.0-0.0-5.0-10.0-15.0-20.0/imagenet_q64/latest.pth \
  --exp_name imagenet_q64_gain --batch 16 --snr_db_min -5 --snr_db_max 20
```

## 3. Gain 보정 관측값을 이용한 diffusion posterior sampling

Classification Q inference는 **`inference_orthogonal_power.py`**로 통합했습니다.
Oracle과 gain estimator가 같은 송신·복원 경로를 사용하며 기본값은
**AWGN(`h=1`), DDRM, uniform의 `Q=I`**입니다. Allocated는 지정한
checkpoint의 학습 Q와 allocator를 그대로 사용합니다. Checkpoint를 한 번
읽어 Q·allocator·classifier에 같은 파일의 상태를 사용합니다.

`--receiver_mode oracle`은 실제 gain을 사용하고, `gain_estimator`는 추정 gain을
사용합니다. 수신 모드를 생략하면 estimator checkpoint가 있을 때 gain estimator,
없을 때 oracle을 선택합니다. `--channel_model rayleigh`와
`--restoration_method dps`도 명시적으로 선택하실 수 있습니다.
Classification의 `power_allocation.png`는 전략별 실제/추정 SD1.5 diffusion
timestep 맵을 표시합니다. 실제 noise 분산은 `N0/(abs(h)^2*a_b^2)`, 추정 noise
분산은 수신기의 추정 precision의 역수로 계산하며, 기존 SD1.5 변환을 사용합니다.
두 열 모두 0~999 색상 범위를 공유하고 셀에는 블록 ID와 정수 timestep을 표시합니다.
Oracle 수신의 오른쪽 열은 `Receiver timestep (oracle)`이며 실제 맵과 같습니다.
범위 초과 noise는 표시할 때만 999로 제한합니다. 원래 precision과 복원 계산은
유지하며, `true_timestep`과 `estimated_timestep`을 JSON과 baseband 파일에도 저장합니다.
아래는 기존 DPS 설정을 명시한 예제입니다.

```bash
python inference_orthogonal_power.py \
  --checkpoint checkpoints_power/ImageNet/snr_-5.0-0.0-5.0-10.0-15.0-20.0/imagenet_q64/latest.pth \
  --gain_estimator_checkpoint checkpoints_gain_estimator/ImageNet/imagenet_q64_gain/latest.pth \
  --data_path /dataset/ImageNet --sample_index 0 --snr_db 5 \
  --receiver_mode gain_estimator --channel_model awgn --restoration_method dps \
  --diffusion_steps 50 --guidance_scale 0.1 --guidance_max_rms 1.0
```

Gain estimator 수신에서는
`U_tilde_b = Y_b / (h a_hat_b)`와
`lambda_hat_b = |h|² a_hat_b² / N0`를 계산하고, 매 diffusion step에서
다음 관측 손실을 사용합니다.
수신 보정에는 복소 h 자체를 사용하여 위상과 크기를 모두 제거합니다.

```text
L_obs = 0.5 * sum_b lambda_hat_b * ||[Q^T P_p(z0_hat)]_b - U_tilde_b||²
```

최대 timestep 999의 독립 Gaussian latent에서 시작하며, 수신 latent를
초기값으로 사용하지 않습니다. 모든 DDIM step에서 `z0_hat` 예측과 UNet을
통과해 현재 `z_t`까지 역전파하고,
다음과 같이 보정합니다. `g_i`는 이미지별 모든 실수 성분에 대한 gradient이고,
`k_(i+1)`은 다음 scheduled timestep입니다.

```text
g_i = grad_(z_(k_i))(L_obs)
r_i = sqrt(mean(g_i^2))
g_clipped = g_i * min(1, 1 / max(r_i, 1e-12))
eta_i = guidance_scale * sqrt(alpha_bar[k_(i+1)])
z_(k_(i+1)) = DDIM_step(z_(k_i)) - eta_i * g_clipped
```

gradient의 RMS 상한은 기본 1입니다. 마지막 step은 scheduler의
`final_alpha_cumprod`를 다음 상태의 alpha로 사용합니다.
원시 수신값 Y와 직접 비교하지 않습니다. 관측 손실은 사용자 수식의
`lambda = |h|² a_hat² / N0` 정의를 사용합니다. packed real I/Q 성분 하나의
잡음 분산은 `N0`이며, gain 보정 후에는 `1/lambda`입니다.
잡음은 `sqrt(N0) * (epsilon_real + 1j * epsilon_imag)`에 해당하며,
복소 잡음 전체의 평균 에너지는 `2*N0`입니다. 새 estimator의 입력은
채널 보정 신호 `R_h`와 `log(nu)`, `log(P)`입니다.

`--guidance_max_rms`는 이미지별 관측 gradient의 RMS 상한입니다.
0으로 지정하면 clipping을 끕니다. 이전 옵션 이름 `--guidance_max_norm`도
동일한 RMS 상한을 지정하는 별칭으로 지원합니다.
`--guidance_scale`은 위 식의 기본 step size `eta_0`이고, 각 step에서는
다음 timestep의 `sqrt(alpha_bar)`를 곱합니다. `--posterior_eta`로 DDIM stochasticity를
조절할 수 있으며 기본값은 0입니다. 이 값들과 guidance scale은 검증 데이터로
조정해 주시면 됩니다. `--receiver_mode oracle`은 gain 추정 오차를 제외한 비교용입니다.

allocated와 uniform 전략에 같은 복소 h, 같은 채널 잡음과 같은 diffusion 초기 잡음을
사용합니다. `result.json`에 직접 수신/복원 후 분류 결과, gain, precision,
step별 관측 손실을 저장하고, `*_baseband.pt`에 Q, raw 수신값, 보정 관측값,
복원 latent와 실제 복소 h를 저장합니다. `visualize_power_allocation.py`는 변환 블록별
전력을 `transformed_block_power.png`에 표시합니다.

기존 Q 없는 checkpoint의 추론은 기존 경로를 사용합니다. 기존 학습을 재개하려면
`--no-orthogonal_transform` 및 checkpoint와 동일한 pilot/SNR conditioning
옵션을 지정해 주셔야 합니다. Q가 추가된 실험은 새로 학습해 주셔야 합니다.
기존 `train_timestep_estimator.py`의 checkpoint와 새 gain estimator는 서로 호환되지 않습니다.
이전 고정 h/AWGN의 Q checkpoint와 gain estimator도 Rayleigh 경로에서는
재학습해 주셔야 합니다. 채널 모델을 checkpoint에 기록하고 gain estimator 버전을
5로 구분하여 이전 설정이 섞이지 않도록 검사합니다. 구조명은 각각
`block_mlp_v4`, `block_conv1d_global_mlp_v4`로 기록합니다.
이전 2D CNN allocator/estimator checkpoint는 새 구조와 호환되지 않습니다.
이전 `N0/2` 잡음 설정의
Q/allocator와 gain estimator도 새로운 `N0` 설정으로 재학습해 주셔야 합니다. Rayleigh 경로에서
고정 `--channel_gain_abs`는 사용하지 않습니다.
이전 실수 성분당 송신 전력 0.5 설정의 Q/allocator와 gain estimator도
`P=1` 설정으로 새로 학습해 주셔야 합니다. 추론은 수신 및 DPS 복원 이미지
확인용이며, 전체 ImageNet 평가를 수행하지 않습니다.

### DDRM 수신 분기: Oracle Gain부터 비교하기

통합 inference의 기본 복원 방식은 DDRM입니다.
`--receiver_mode oracle`로 실제 gain을 사용하는 복원을 확인하실 수 있습니다.
Checkpoint에 classifier가 있으면 직접 복원 및 diffusion 복원 뒤의 분류 결과도
함께 기록합니다. Oracle 모드는 gain estimator를 불러오지 않습니다.

```bash
python inference_orthogonal_power.py \
  --checkpoint checkpoints_power/ImageNet/snr_-5.0-0.0-5.0-10.0/imagenet_1111/latest.pth \
  --data_path /dataset/ImageNet --sample_index 0 --snr_db 10 \
  --receiver_mode oracle --channel_model awgn \
  --restoration_method ddrm --diffusion_steps 50 \
  --ddrm_eta 0.85 --ddrm_eta_b 1.0
```

기본 AWGN 모드는 h=1로 테스트합니다. `--channel_model rayleigh`를 지정하면
실제 복소 h를 보정합니다. 같은 seed로 `--restoration_method dps`를 실행하면
동일한 VAE sample, h, 채널 잡음으로 비교하실 수 있습니다. 두 복원 방식은
각자의 초기화 및 sampling 규칙을 사용합니다. 각 실행에서 uniform/no-Q와
allocated/Q 전략에는 같은 diffusion 난수 스트림을 재사용합니다.

[DDRM 원 논문](https://arxiv.org/abs/2201.11793)의 Eq. (7), (8)을
`ddrm_sampling.OrthogonalDDRM`에 구현했습니다. 기존 notation과의 대응은
`V=Q`, `y_bar=U_tilde`, `sigma_y/s_b=1/sqrt(lambda_b)`입니다.
하나의 Q를 블록 내 모든 coefficient에 적용하므로 추가 SVD나 singular value
정렬이 필요하지 않습니다. 아래에서 `sigma`는 **다음** scheduled timestep의
noise scale이며 `Xi`는 매 단계 독립 표준 Gaussian입니다.

```text
sigma_k^2 = (1-alpha_bar[k]) / alpha_bar[k]
W_T,b = U_tilde_b + sqrt(sigma_T^2 - 1/lambda_b) * Xi_T,b
s_T = sqrt(alpha_bar[T]) * P_p^-1(Q W_T)

z0_hat = (s_k - sqrt(1-alpha_bar[k])*epsilon_theta(s_k,k)) / sqrt(alpha_bar[k])
U0_hat = Q^T P_p(z0_hat)

A: sigma^2 < 1/lambda_b
W_next,b = U0_hat_b + sqrt(1-eta^2)*sigma*sqrt(lambda_b)*(U_tilde_b-U0_hat_b)
          + eta*sigma*Xi_b
B: sigma^2 >= 1/lambda_b
W_next,b = (1-eta_B)*U0_hat_b + eta_B*U_tilde_b
          + sqrt(sigma^2-eta_B^2/lambda_b)*Xi_b

s_next = sqrt(alpha_bar[next]) * P_p^-1(Q W_next)
```

UNet은 원래 spatial VP latent인 `s_k`에서 실행합니다. 블록별 갱신만 Q 좌표에서
수행하며 역전파나 DPS guidance는 사용하지 않습니다. DDIM `scheduler.step` 대신
실제 다음 scheduled timestep의 alpha로 갱신하고, 마지막에는 scheduler의
`final_alpha_cumprod`와 관계없이 `alpha=1`, `sigma=0`으로 clean latent를 반환합니다.
`epsilon`, `v_prediction`, `sample` prediction type을 지원합니다.

`--ddrm_eta`는 Case A의 sampling noise, `--ddrm_eta_b`는 Case B의 관측 혼합
계수입니다. 둘 다 [0,1] 범위이며 기본값은 각각 0.85와 1.0입니다.
`--guidance_scale`, `--guidance_max_rms`, `--posterior_eta`는 DPS용 옵션이며
DDRM 갱신에는 사용하지 않습니다.

DDRM에서는 `--max_timestep`을 생략하면 **999**가 적용됩니다. 기존 allocator
출력에 `a_min^2=N0/(abs(h)^2*sigma_999^2)` 하한을 먼저 확보하고 남은 에너지를
배분합니다. `--max_timestep 650`처럼 직접 지정하면 해당 값으로 덮어씁니다.
이 옵션은 송신 gain 하한의 기준이며 DDRM의 시작 timestep과 sampling 횟수는
기존 설정을 사용합니다. DPS에서는 이 하한을 사용하시려면 옵션을 명시하셔야 합니다.

논문 Eq. (7)의 초기화에는 `sigma_T^2 >= 1/lambda_b`가 필요합니다.
현재 구현은 양의 finite precision을 사용하는 full-rank 수신기입니다.
**Oracle과 gain estimator 수신 모두** 위 조건을 넘는 블록도 timestep 999에서
계속 복원합니다. 초기화에 추가하는 잡음 분산만 아래처럼 0으로 제한합니다.

```text
initial_variance_b = max(sigma_T^2 - 1/lambda_b, 0)
W_T,b = U_tilde_b + sqrt(initial_variance_b) * Xi_b
```

초과 블록은 `W_T,b=U_tilde_b`로 시작합니다. 관측값과 수신기에 전달된 precision은 그대로
유지하고, 이후 Eq. (8)에서는 원래 `lambda_b`를 사용합니다. 이 처리는 초기
분포가 timestep 999와 정확히 일치하지 않는 실험용 근사입니다. Gain estimator
수신에서도 추정 gain과 추정 precision은 수정하지 않습니다. 모델이 log-gain을
출력하므로 이를 noise로 환산한 값은 timestep 999의 최대 noise를 넘을 수 있습니다.
DDRM의 기본 `max_timestep=999` 송신 gain 하한도 유지하지만 전력 예산이 부족한
경우에는 위 초기화 근사로 진행합니다. Uniform/no-Q와 allocated/Q에 동일한
초기화 정책을 적용하며 이미지를 건너뛰지 않습니다. 공통 sampler 구성을 사용하므로
inference와 MaxViT eval에 동일하게 적용됩니다.

oracle 실행은 `*_oracle_ddrm.png`, `ddrm_vs_input`, `ddrm_latent_mse`를 저장하고,
`result.json`의 `restoration_method`와 `diffusion`에 실제 방식 및 초기화 설정,
단계별 sigma, Case A/B 블록 수와 관측 손실을 기록합니다. 기본 저장 경로도
DDRM 및 eta/eta_B 값을 포함하므로 DPS 결과와 구분됩니다.
`diffusion.initialization_policy`는 oracle과 추정 gain 모두 `clip`입니다.
`initialization_clipped_blocks`는 이미지별 초과 블록 수이며,
`initialization_clipped_images`와 `initialization_clipped_block_count`도 기록합니다.

`inference_power.py`의 Q 분기도 통합 구현을 호출합니다.
`--receiver_mode oracle`이면 실제 gain을, `gain_estimator`이면 기존 추정 gain과
그 precision을 사용합니다. 추정 gain을 고정한 DDRM은 근사 관측 모델을 사용하며,
추정 오차에 대한 추가 보정은 구현하지 않았습니다. Q 없는 legacy checkpoint에서는
DDRM 옵션을 오류로 안내합니다.

현재 classification gain estimator를 DDRM 복원에 연결하는 명령은 다음과 같습니다.

```bash
python inference_orthogonal_power.py \
  --checkpoint checkpoints_power/ImageNet/snr_-5.0-0.0-5.0-10.0-15.0-20.0/imagenet_1106_amp/latest.pth \
  --gain_estimator_checkpoint checkpoints_gain_estimator/ImageNet/imagenet_gain_estimator/latest.pth \
  --receiver_mode gain_estimator --restoration_method ddrm \
  --channel_model awgn \
  --sample_index 0 --snr_db 10 --diffusion_steps 25 \
  --max_timestep 999 --seed 0 \
  --output_dir results/gain_estimator_inference
```

`Y/h`에서 추정한 `a_hat`으로 관측값 `Y/(h*a_hat)`과 precision
`|h|^2*a_hat^2/N0`를 계산하여 DDRM에 전달합니다. 시각화의 timestep은
잡음 수준을 설명하는 환산값이며, DDRM은 timestep 999에서 전체 schedule을
실행합니다. 결과 폴더의 `comparison.png`는 원본·allocated·uniform 복원을,
`reconstructed.png`는 allocated 복원을 표시합니다. 직접 gain 보정 후 VAE만
decode한 결과는 `direct_reconstructed.png`입니다. `result.json`과
`allocated_baseband.pt`에는 사용한 추정 gain, precision 및 복원 정보가 저장됩니다.

Oracle과 비교할 때는 같은 checkpoint·이미지·SNR·채널·seed·sampling 옵션을
사용하고 `--receiver_mode`만 바꾸시면 됩니다. VAE, 채널과 diffusion 난수
스트림이 두 수신 모드에서 공통입니다. Uniform은 `Q=I`와 공통 송신 gain을,
allocated는 checkpoint의 Q와 블록별 gain을 사용합니다.

Uniform의 수신 방식은 `--uniform_receiver_mode oracle|gain_estimator`로
따로 지정하실 수 있습니다. **Uniform의 기본값은 oracle**입니다.
아래처럼 allocated에 gain estimator를 지정해도 uniform은 oracle gain으로 복원합니다.

```bash
--receiver_mode gain_estimator
```

반대 비교는 `--receiver_mode oracle --uniform_receiver_mode gain_estimator`이며,
어느 한쪽이라도 gain estimator를 사용하면 `--gain_estimator_checkpoint`가
필요합니다. 모델은 한 번만 로드하며, 송신 신호·h·채널 잡음·diffusion 난수는
기존처럼 공유합니다. DDRM의 oracle 초기화 정책과 추정 gain 초기화 정책은
각 전략의 수신 모드에 맞게 적용합니다.
실제 수신 모드는 콘솔 PSNR/LPIPS, 비교 이미지의 제목, JSON과 baseband 파일에
기록합니다. 두 전략의 수신 모드가 다르면 기본 저장 폴더 이름에도
`_uniform_<mode>`가 추가됩니다.

기본 저장 위치는 두 모드 모두 checkpoint 폴더 아래
`inference/sample_<index>_snr_<SNR>_seed_<seed>_steps_<K>_<receiver>_.../`입니다.
`--output_dir`를 주면 해당 경로를 사용합니다. 기존 `oracle_gain_inference/`와
`results/gain_estimator_inference/`의 저장 결과는 그대로 유지합니다.
`inference_oracle_gain.py`는 중복된 실행 구현을 없애고 oracle 모드를 기본으로
호출하는 호환용 진입점으로 남겼습니다. 공유 helper import도 계속 지원합니다.
`inference_power.py`는 Q가 없는 기존 checkpoint의 실행 경로를 유지합니다.

MaxViT 평가도 동일한 선택 옵션을 지원합니다.

```bash
python evaluate_power_maxvit.py \
  --checkpoint checkpoints_power/ImageNet/snr_-5.0-0.0-5.0-10.0/imagenet_1111/latest.pth \
  --data_path /dataset/ImageNet --snr_db 10 --max_samples 100 \
  --batch_size 4 --restoration_method ddrm --diffusion_steps 50 \
  --ddrm_eta 0.85 --ddrm_eta_b 1.0
```

평가 결과의 `reconstruction`은 `ddrm`으로, 기본 파일명은
`maxvit_ddrm_rayleigh_snr_<SNR>_steps_<K>_eta_<eta>_etab_<eta_B>_seed_<seed>_balanced_n<N>_maxt_999.json`으로
기록됩니다. Oracle 평가에서는 초기화 한도를 넘는 블록의 추가 noise를 0으로 두고
진행합니다. 중간 로그에 전략별 누적 이미지·블록 수를 출력하고, 각 SNR 결과의
`ddrm_initialization_clipped`에 uniform/allocated별 합계를 저장합니다.

## Q checkpoint의 MaxViT 최종 평가와 LPIPS

`evaluate_power_maxvit.py`는 최근 Q+allocator checkpoint를 사용해 다음 세 가지를
같은 ImageNet validation 이미지에서 평가합니다.

- `vae`: VAE posterior sample을 바로 decode합니다.
- `uniform`: Q 없이 하나의 공통 송신 gain을 적용하고 Oracle Gain으로 DPS 복원합니다.
- `allocated`: 학습한 Q와 allocator로 전송하고 Oracle Gain으로 DPS 복원합니다.

이미지마다 `h=(randn(B)+1j*randn(B))/sqrt(2)`를 새로 생성하며 64개 블록에서
같은 h를 사용합니다. Uniform과 allocated는 동일한 h, AWGN, DPS 초기 잡음을 공유하고,
실수 성분당 평균 송신 전력 1, 잡음 분산 N0를 사용합니다. 기본 수신 모드는
두 방식 모두 oracle이며, 학습에 사용한 latent classifier는 평가에 사용하지 않습니다.

Allocated 수신에 gain estimator를 사용하시려면 기존 명령에
`--receiver_mode gain_estimator --gain_estimator_checkpoint <estimator checkpoint>`를
추가하시면 됩니다. Uniform은 oracle을 유지합니다. Estimator는 실제 제공된 Q와
allocator를 사용하며, 학습 당시 allocator fingerprint와 다르더라도 차단하지 않습니다.
Allocated 복원에는 추정 gain으로 equalize한 관측과 추정 precision을 사용합니다.
DDRM 초기화 정책은 uniform과 allocated 모두 `clip`을 사용하며,
초과 블록 수를 전략별로 기록합니다. 기본 결과 파일에는 `_receiver_gain_estimator_uniform_oracle`을
붙여 oracle 결과와 구분하고, JSON에 수신 모드와 estimator 경로/설정을 기록합니다.

정확도는 사전학습 RGB MaxViT-Large와 공통 `power_utils.maxvit_input`의
235 bicubic resize / 224 center crop / ImageNet 정규화로 측정합니다.
LPIPS는 공식 VGG16 v0.1 모델을 사용하고 원본 해상도의 RGB를 [-1,1]로 변환해
계산합니다. 각 방식의 Top-1, Top-5, 평균 PSNR, 평균 LPIPS를 저장합니다.

```bash
python -m pip install -r requirements_eval.txt

python evaluate_power_maxvit.py \
  --checkpoint checkpoints_power/ImageNet/snr_-5.0-0.0-5.0-10.0-15.0-20.0/imagenet_1106_amp/latest.pth \
  --data_path /dataset/ImageNet --snr_db 10 \
  --batch_size 4 --diffusion_steps 25 --guidance_scale 0.1
```

기본값은 `--max_samples 5000`이며, ImageNet validation의 1,000개 클래스에서
각각 5장씩 중복 없이 무작위로 선택합니다. 선택에는 `--seed`를 사용하는 별도
난수 생성기를 사용하므로 같은 seed와 dataset에서는 같은 이미지가 선택됩니다.
VAE / uniform / allocated와 모든 SNR은 한 번 선택한 동일한 이미지 목록을
같은 순서로 평가합니다. Dataset 앞부분 5,000장을 잘라서 사용하지 않습니다.

전체 50,000장을 평가하시려면 `--max_samples 50000`을 지정해 주시면 됩니다.
다른 개수를 지정해도 클래스별 선택 수가 최대 1장만 차이 나도록 배분합니다.
예를 들어 `--max_samples 100`은 무작위로 선택한 100개 클래스에서 1장씩
평가하는 짧은 점검용입니다. 특정 클래스의 이미지가 필요한 할당량보다 적으면
불균등하게 채우지 않고 오류를 표시합니다.

`--snr_db 0 5 10`처럼 여러 값을 지정하면 각 SNR의 세 가지 결과를 각각 별도
파일에 저장합니다. 결과는 checkpoint 폴더의
`evaluation/maxvit_dps_rayleigh_snr_<SNR>_*_balanced_n<N>.json`에 저장합니다.
여러 SNR과 `--output_path`를 함께 지정하면 지정한 파일명에 `_snr_<SNR>`을 붙입니다.
각 JSON의 `sample_selection`에는 선택 seed, 원본 dataset index, 상대 이미지 경로와
클래스별 장수를 기록합니다. 실행 중인 평가 프로세스에는 소스 변경이 반영되지
않으므로 변경된 선택 방식을 사용하시려면 평가를 새로 실행해 주셔야 합니다.

평가는 단일 GPU에서 실행합니다. 첫 배치와 `--log_interval`(기본 10) 배치마다
VAE / uniform / allocated의 누적 Top-1, Top-5, PSNR, LPIPS를 콘솔에 출력합니다.
매 배치 확인하려면 `--log_interval 1`을 지정해 주시면 됩니다.

현재 이미지 확인용 `inference_orthogonal_power.py`, `inference_power.py`, `inference_oracle_gain.py`,
`inference_legacy_power_dps.py`도 LPIPS를 기본으로 측정합니다. `--no-lpips`로
해당 측정만 끌 수 있습니다. `inference_power.py`의 Q 경로에도 동일하게 적용됩니다.

Inference와 `evaluate_power_maxvit.py`에서 `--min_gain 0.3`을 추가하면
allocated 송신 진폭에만 `a_b >= 0.3` 하한을 적용합니다. 각 블록의
`0.3^2 * E_b` 전력을 먼저 확보하고, 남은 전력을 기존 softmax 비율로
배분하므로 총 송신 에너지는 유지됩니다. 수신 gain 보정과 precision도
변경된 실제 gain을 사용합니다. 기본값 `0`은 기존 동작이며 학습에는
적용하지 않습니다. 하한만으로 총 에너지 예산을 넘으면 오류를 표시합니다.
기본 저장 경로에는 `_mingain_0.3`을 붙여 하한 없는 결과와 구분합니다.

고정 gain 대신 `--max_timestep 650`을 주면 SD1.5의
`v_max=(1-alpha_bar[650])/alpha_bar[650]` 기준으로 보정 후 잡음 분산을
제한합니다. 현재 Rayleigh/AWGN 링크에서는 각 이미지의 실제 `h`와 `N0`로
`a_min^2=N0/(abs(h)^2*v_max)`를 계산하고 그 전력을 먼저 확보합니다.
`--min_gain`도 지정하면 두 하한 중 큰 값을 사용합니다. 이 옵션은 DPS의
시작 timestep이나 sampling step 수를 변경하지 않습니다. 현재 schedule의
600/650/700은 각각 diffusion SNR 약 -7.18/-8.77/-10.46 dB입니다.
깊은 fading으로 timestep 하한 전력이 총 예산을 넘으면 해당 이미지에는
같은 예산의 균일 gain을 사용합니다. 이 경우 timestep 상한은 보장되지 않으며
`gain_floor_infeasible_images`로 건수를 기록합니다. 평가 중간 로그에도 건수를
출력합니다. Uniform 비교 경로는 그대로입니다. Legacy 링크는 원래의 `N0/2`와
quantizer scaling을 사용하며 이 timestep 기준은 oracle 보정 잡음에 해당합니다.

## Classification gain estimator의 블록별 비교

`visualize_gain_estimator.py`는 학습한 estimator의 실제/추정 gain을
**SD1.5의 보정 후 잡음 timestep으로 환산**하여 64개 Q 블록별로 비교합니다. VAE, 고정된
Q/allocator와 estimator만 로드하므로 UNet, diffusion, classifier, LPIPS를
실행하지 않습니다. 그림 저장에는 `matplotlib`이 필요합니다.

```bash
python visualize_gain_estimator.py \
  --gain_estimator_checkpoint checkpoints_gain_estimator/ImageNet/imagenet_gain_estimator/latest.pth \
  --channel_model awgn \
  --sample_index 0 --snr_db -5 0 5 10 15 20 --seed 0
```

Allocator checkpoint는 estimator에 저장된 경로를 사용합니다. 시각화에서는
학습 당시와 Q/allocator fingerprint가 달라도 현재 allocator가 실제로 보낸
gain을 정답으로 삼아 비교합니다. 학습 당시/현재 fingerprint와 일치 여부는
`result.json`의 `allocator_matches_training` 등에 기록합니다. 경로를 옮기셨다면
`--allocator_checkpoint <송신기/latest.pth>`를 추가하시면 됩니다.
입력은 ImageNet val의 `--sample_index`로 선택하며 `--data_path`로 dataset
경로를 변경하실 수 있습니다. 원하는 이미지에는 `--input_image <파일>`을
사용하시면 됩니다. 입력 전처리는 기존 oracle inference와 같은 resize/center crop입니다.

시각화의 기본 채널은 **AWGN이며 `h=1+0j`로 고정**합니다. 따라서 표시되는
실효 SNR은 지정한 SNR과 같습니다. Rayleigh 비교를 원하시면
`--channel_model rayleigh`를 명시하시면 됩니다. 실제 사용한 채널과 학습 시
채널은 결과 JSON에 각각 기록하고, 그림 제목에는 실제 `|h|`를 표시합니다.
SNR을 여러 개 지정해도 같은 이미지, VAE posterior sample, 이미지 전체에
공유하는 복소 h와 표준 정규 잡음 epsilon을 유지합니다. SNR마다 N0와
allocator의 CSI 조건만 바꿔 실제 gain과 추정 gain을 비교합니다.
잡음은 `sqrt(N0)*epsilon`으로 크기를 변경합니다.

**맵은 원본 이미지 위에 겹치지 않습니다.** Q로 섞인 변환 후 블록 ID
0~63을 8×8로 배치합니다. 각 SNR 폴더에는 다음 결과가 저장됩니다.

- `timestep_heatmaps.png`: 실제 gain 기준 timestep, 추정 gain 기준 timestep, `추정 t - 실제 t`를 나란히 표시합니다. 모든 SNR에서 같은 0~999 색상 척도를 사용합니다.
- `timestep_by_block.png`: 블록 ID별 실제/추정 timestep 곡선과 timestep 오차 막대그래프를 표시합니다.
- `blocks.csv`: 64개 블록의 timestep·오차, gain, log gain, 상대오차, 실제/추정 precision, 전력 비율을 저장합니다.

각 gain으로 `v=N0/(|h|^2*a_b^2)=1/precision`을 계산하고, SD1.5의
`(1-alpha_bar[t])/alpha_bar[t] >= v`를 처음 만족하는 timestep에 대응시킵니다.
스케줄 범위를 넘으면 0 또는 999로 표시하며 해당 셀에 `*`를 붙입니다.
원래 잡음 분산, 범위 초과 여부와 블록 수는 CSV/JSON에도 남깁니다.
이 timestep은 블록별 보정 후 잡음 수준이며 DDRM의 시작 timestep은 변경하지 않습니다.
Estimator 학습은 계속 log gain 회귀를 사용합니다.

기존 gain 맵은 `--display gain`으로 선택하실 수 있습니다. 이때는
`gain_heatmaps.png`, `gain_by_block.png`를 저장합니다. 기본값은 `--display timestep`입니다.

상위 폴더에는 timestep MAE·RMSE·최대 오차를 비교하는 `snr_comparison.png`,
수치와 h·설정 SNR·실효 SNR을 담은
`result.json`, 재현에 필요한 `observations.pt`, 입력 확인용 `input.png`가 저장됩니다.
그림의 gain은 수신기에 실제로 적용하는 `exp(clamp(predicted_log_gain,-20,20))`입니다.
원래 log 예측값과 clamp된 블록 수를 별도로 기록합니다.
`log_gain_mse`는 원래 log 예측으로 계산하며, `gain_relative_mae`와
`precision_relative_mae`는 실제 적용 gain 기준의 상대오차입니다.
`training_gain_relative_mae`는 학습 로그와 같은 계산식을 사용합니다.
`timestep_mae`, `timestep_rmse`, `timestep_max_error`는 환산한 timestep 차이를 step 단위로 표시합니다.

기본 저장 위치는 estimator checkpoint 폴더 아래 `visualizations/`이며,
`--output_dir`로 지정하실 수 있습니다. 기본값은 gain estimator 학습과 같은
**gain floor 없는 송신**입니다. DDRM inference의 송신 조건을 확인하시려면
`--max_timestep 999`를 명시하시면 됩니다. `--min_gain`도 지원합니다.
이 옵션들은 송신 gain과 정답에도 함께 반영되며 전력 예산을 유지합니다.

### Gain estimator를 1,000장으로 평가하기

같은 스크립트에 `--num_samples 1000`을 지정하시면 ImageNet val의
1,000개 클래스에서 각각 1장씩 무작위로 선택해 평가합니다. 선택에는
`--seed`를 사용하며 중복 없이 같은 이미지 목록을 모든 SNR에서 사용합니다.
`--num_samples`의 별칭으로 `--max_samples`도 지원합니다.

```bash
python visualize_gain_estimator.py \
  --gain_estimator_checkpoint checkpoints_gain_estimator/ImageNet/imagenet_gain_estimator/latest.pth \
  --num_samples 1000 --batch_size 4 --num_workers 4 \
  --channel_model awgn --snr_db -5 0 5 10 15 20 --seed 0 \
  --output_dir results/gain_estimator_check_1000
```

VAE·Q/allocator·estimator를 한 번만 로드하며, VAE encode는 이미지당 한 번입니다.
각 배치의 latent·h·표준 잡음을 모든 SNR에 재사용합니다. 기본 AWGN에서는
모든 이미지의 h가 정확히 `1+0j`입니다. Diffusion, classifier, LPIPS는 실행하지
않고 이미지별 그림도 만들지 않습니다. 진행 상황은 첫 배치 및 기본 10배치마다
표시하며, `--log_interval`로 간격을 조절하실 수 있습니다.

전체 timestep MAE·RMSE·p90·최대 오차와 gain 상대오차는 **1,000×64개의
이미지/블록 쌍**에서 계산합니다. RMSE는 이미지별 RMSE의 평균이 아닙니다.
각 SNR의 8×8 맵은 블록별 평균 실제/추정 timestep과 **평균 절대 timestep 오차**를
보여줍니다. 양수·음수 오차가 상쇄된 평균 차이를 정확도처럼 표시하지 않습니다.
`--display gain`에서도 같은 방식으로 평균 gain과 평균 절대 상대오차를 표시합니다.

배치 평가 결과는 다음과 같이 저장합니다.

- `result.json`: 전체 통계, 블록별 집계, 선택한 dataset index·경로·클래스와 평가 시간입니다.
- `per_image_metrics.csv`: 각 이미지/SNR의 gain·timestep 오차와 실제 사용한 `|h|`·실효 SNR입니다.
- `snr_<SNR>/blocks.csv`, `timestep_heatmaps.png`, `timestep_by_block.png`: 64개 블록의 평균값·MAE·RMSE·스케줄 초과 건수와 집계 그림입니다.
- `snr_comparison.png`: SNR별 전체 MAE·RMSE·최대 오차 비교입니다.
- `sample_values.pt`: 선택된 이미지 순서대로 실제 gain·예측 log gain·실제/추정 precision과 h를 저장합니다.

기존 단일 이미지 모드는 `--num_samples 1`이 기본이며, `--sample_index`로
이미지를 고르는 방식도 유지합니다. 여러 장 평가에는 `--data_path`로
ImageNet 경로를 지정하실 수 있습니다. 생략하면 allocator checkpoint의 경로를 사용합니다.

## 검증

```bash
python -m unittest discover -s tests -p 'test_orthogonal_power.py' -v
python -m unittest discover -s tests -p 'test_ddrm_sampling.py' -v
python -m unittest discover -s tests -p 'test_evaluate_power_sampling.py' -v
python -m unittest discover -s tests -p 'test_gain_estimator_visualization.py' -v
python -m unittest discover -s tests -p 'test_segmentation_transport.py' -v
torchrun --standalone --nproc_per_node=2 tests/ddp_smoke_orthogonal_power.py
torchrun --standalone --nproc_per_node=2 tests/ddp_smoke_segmentation_power.py
```

직교성, 역변환, 에너지 제약, Q/allocator/classifier의 gradient, gain·채널 보정,
precision, 관측 손실의 해석적 gradient, estimator의 송신기 고정,
checkpoint 호환성과 DPS의 UNet Jacobian을 검사합니다.
추가로 Rayleigh 샘플의 분포와 재현성, 공통 h의 복소 위상 회전과 보정,
estimator의 위상 불변성, perfect CSI의 allocator 전달을 검사합니다.
gradient RMS 제한과 timestep별 guidance schedule, 마지막 step의 alpha,
실수 성분당 `P=1` 전력 및 설정 SNR과의 일치도 검사합니다.
VAE posterior sampling의 seed 재현성과 gradient 차단도 검사합니다.
기본 네트워크의 파라미터 수와 Section VI-B의 6개 통계량 공식도 검사합니다.
DDRM 검증은 Eq. (7)의 spatial VP 초기화, Eq. (8) 두 분기의 경험적 평균·분산,
noise 경계에서의 분기, clean terminal state, prediction type, seed 재현성,
초기화 조건 위반과 oracle 이미지/JSON/latent 저장 경로를 검사합니다.
DDP smoke test는 외부 가중치 없이 작은 VAE/classifier를 사용하여 두 프로세스에서
Q, allocator, classifier 및 gain estimator의 파라미터가 동기화되는지 확인합니다.
기본 backend는 CPU/gloo이며, `--device cuda`로 GPU/NCCL 경로를 검사할 수 있습니다.

## Cityscapes segmentation: Q512 + Rayleigh + DDRM

데이터셋, train/val 분할, 19개 클래스와 ignore index 255를 유지합니다.
입력은 기존 1024×2048이고 SD1.5 VAE latent는 4×128×256입니다.
기본 **latent 패치 8×8**에서 공간 격자는 16×32, 블록 수는 512,
블록당 실수 성분은 256개이며 **Q는 512×512 정방행렬**입니다.
공간 격자가 직사각형이어도 Q의 두 축은 모두 블록 ID이므로 정방행렬입니다.

`segmentation_transport.py`는 ImageNet과 같은 Q 및 block-MLP allocator를
사용합니다. 송신은 `U=Q^T P(z)`이고 블록별 진폭을 곱한 뒤 Rayleigh 채널을
통과합니다. `h~CN(0,1)`은 이미지마다 한 번 샘플링하여 모든 블록에 공유합니다.
실수 성분당 평균 송신 전력 `P=1`, 잡음 분산 `N0=10^(-SNR/10)`을 사용하며,
전체 송신 에너지 예산은 `4*128*256*P=131072`입니다.
Allocator에는 perfect CSI로 계산한 `log_gamma=ln(|h|^2*P/N0)`를 전달합니다.

학습에서는 frozen VAE 뒤의 **Q·allocator·latent SegFormer-B0**를 pixel CE로
함께 갱신합니다. Oracle gain 보정과 Q 역변환을 거친 수신 latent를 SegFormer에
전달합니다. 학습 및 학습 중 validation에는 diffusion을 실행하지 않습니다.
학습 중 기록하는 mIoU는 이 latent SegFormer의 직접 수신 성능입니다.

추론과 최종 평가는 **DDRM이 기본값**입니다. Oracle 보정 관측값과 실제
`lambda_b=|h|^2*a_b^2/N0`를 전달하고, 블록 공간의 DDRM 갱신과 공간 latent의
SD1.5 UNet 호출 사이를 Q로 변환합니다. 기존 DPS도
`--restoration_method dps`로 선택하실 수 있습니다.
최종 평가는 복원 latent를 VAE decode한 RGB에 기존 pretrained Mask2Former를
적용하여 pixel accuracy와 mIoU를 계산합니다. 학습한 latent SegFormer는
이 최종 평가에 사용하지 않습니다.

Uniform은 Q 없이 공간 latent에 공통 gain을 적용합니다. Allocated는 학습한
Q와 블록별 gain을 사용합니다. 두 방식의 전력 예산은 같고, **같은 이미지·VAE
posterior sample·h·채널 잡음·diffusion 난수**로 비교합니다.
`uncompressed`는 채널을 거치지 않은 VAE encode/decode 기준선입니다.

새 실험 학습과 최종 평가는 다음과 같이 실행하실 수 있습니다.

```bash
python train_segmentation_power_allocator.py \
  --exp_name cityscapes_q512_rayleigh \
  --data_path /data/bsy --batch 1 --amp \
  --patch_height 8 --patch_width 8 --max_timestep 999

python eval_segmentation_power_allocator.py \
  --checkpoint checkpoints_power/Cityscapes/SegFormer-B0/snr_uniform/cityscapes_q512_rayleigh/latest.pth \
  --data_path /data/bsy --snr_db 0 --batch 1 --amp \
  --restoration_method ddrm --diffusion_steps 25
```

기본 평가는 Cityscapes val 전체입니다. 짧게 확인하시려면 `--max_samples 10`을
추가하시면 됩니다. `--strategy all`이 기본이며 uniform/allocated/uncompressed를
모두 평가합니다. 단일 이미지 복원과 배정 시각화는 다음과 같습니다.

```bash
python inference_segmentation_power.py \
  --checkpoint checkpoints_power/Cityscapes/SegFormer-B0/snr_uniform/cityscapes_q512_rayleigh/latest.pth \
  --sample_index 0 --snr_db 0 --amp

python visualize_segmentation_power.py \
  --checkpoint checkpoints_power/Cityscapes/SegFormer-B0/snr_uniform/cityscapes_q512_rayleigh/latest.pth \
  --sample_index 0 --snr_db 0
```

시각화 명령은 diffusion 없이 전력 배정과 직접 수신 결과만 저장합니다.
Q로 섞인 블록은 원본 이미지의 특정 공간 패치와 대응하지 않으므로,
allocated 전력 그림은 **변환 후 블록 ID** 기준으로 표시합니다.

`--patch_height 16 --patch_width 16`이면 Q128×128,
`--patch_height 32 --patch_width 32`이면 Q32×32로 새로 학습하실 수 있습니다.
추론·평가에서는 checkpoint에 저장된 패치 구성을 자동으로 읽습니다.
재개할 때는 같은 패치 옵션과 `--resume <latest.pth>`를 사용하시면 됩니다.
기존 CNN/AWGN allocator checkpoint와 새 Q allocator는 구조가 다르므로
새 실험이 필요합니다. 기존 checkpoint에서 SegFormer만 가져오시려면
`--segmenter_checkpoint <기존/latest.pth>`를 지정하시면 됩니다.
기존 AWGN 동작은 네 segmentation 명령에 `--legacy_awgn`을 추가하시면 됩니다.

`--max_timestep` 기본값은 999이며 송신 gain의 noise floor 기준입니다.
DDRM은 여전히 timestep 999에서 시작합니다. Deep fading으로 floor가 전력 예산을
넘으면 동일 예산의 uniform gain으로 fallback하고, 앞서 정의한 oracle 초기화
근사로 진행합니다. 실제 precision은 유지합니다. 이 경우는 결과의
`gain_floor_infeasible_images`와 전략별 DDRM 초기화 초과 이미지·블록 수에 기록합니다.

기본 결과 경로는 checkpoint 폴더 아래입니다.

- 학습: `train.log`, `config.json`, `latest.pth`에 Q/allocator/SegFormer 및 optimizer·scheduler·AMP scaler 상태를 저장합니다.
- 최종 평가: `evaluation/mask2former_ddrm_rayleigh_*.json` 및 같은 이름의 `.log`를 저장합니다. 첫 배치와 기본 20배치마다 중간 결과도 갱신합니다. `--output_path`로 변경하실 수 있습니다.
- 단일 복원: `inference/sample_*/`에 원본·GT·VAE·직접 수신·DDRM 복원 PNG, 비교 그림, `power_allocation.png`, `result.json`, `latents.pt`를 저장합니다.
- 전력 시각화: `allocation/sample_*/`에 저장합니다. 위 두 단일 이미지 명령은 `--output_dir`로 경로를 변경하실 수 있습니다.

Cityscapes 테스트는 기본 Q512의 직교성·역변환, 해상도·패치 구성, 공통 h와
잡음, 전력 예산·precision, Q/allocator/SegFormer gradient, checkpoint·재개,
실제 DDRM의 4×128×256 latent 처리와 deep fading 근사, 모의 최종 평가·이미지
저장을 검사합니다. DDP smoke는 기본 Q512 학습의 두 rank gradient 동기화를 검사합니다.

## Power 전용 코드 구성

Bit allocation의 학습, LUT 압축, bitstream 전송 및 평가 코드는 제거했습니다.
공유하던 SD1.5 noise schedule, latent classifier, MaxViT 전처리와 기존 CNN
power allocator는 `power_utils.py`로 분리했습니다. `inference.py`에는 power
추론에서도 사용하는 diffusion 복원기와 이미지 보조 함수만 남겨 두었습니다.

기존 Q/allocator/classifier 및 optimizer checkpoint의 state-dict 이름과
모델 구조는 유지합니다. ImageNet Q/Rayleigh, DPS/DDRM, gain/timestep estimator,
Cityscapes 및 기존 CNN power 경로는 계속 사용하실 수 있습니다. 저장된
checkpoint, LUT 데이터와 실험 결과는 삭제하지 않았습니다.
