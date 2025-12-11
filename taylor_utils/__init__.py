from typing import Dict
import torch
import math

def derivative_approximation(cache_dic: Dict, current: Dict, feature: torch.Tensor):
    """
    Compute derivative approximation.
    :param cache_dic: Cache dictionary.
    :param current: Current step information.
    """
    difference_distance = current['activated_steps'][-1] - current['activated_steps'][-2]
    # difference_distance = current['activated_times'][-1] - current['activated_times'][-2]

    updated_taylor_factors = {}
    updated_taylor_factors[0] = feature

    for i in range(cache_dic['max_order']):
        if (cache_dic['cache'][-1][current['layer']][current['module']].get(i, None) is not None) and (current['step'] < (current['num_steps'] - cache_dic['first_enhance'] + 1)):
            updated_taylor_factors[i + 1] = (updated_taylor_factors[i] - cache_dic['cache'][-1][current['layer']][current['module']][i]) / difference_distance
        else:
            break
    
    cache_dic['cache'][-1][current['layer']][current['module']] = updated_taylor_factors

def taylor_formula(cache_dic: Dict, current: Dict) -> torch.Tensor: 
    """
    Compute Taylor expansion error.
    :param cache_dic: Cache dictionary.
    :param current: Current step information.
    """
    x = current['step'] - current['activated_steps'][-1]
    # x = current['t'] - current['activated_times'][-1]
    output = 0

    for i in range(len(cache_dic['cache'][-1][current['layer']][current['module']])):
        output += (1 / math.factorial(i)) * cache_dic['cache'][-1][current['layer']][current['module']][i] * (x ** i)
    
    return output

def taylor_cache_init(cache_dic: Dict, current: Dict):
    """
    Initialize Taylor cache and expand storage for different-order derivatives.
    :param cache_dic: Cache dictionary.
    :param current: Current step information.
    """
    if current['step'] == (current['num_steps'] - 1):
        cache_dic['cache'][-1][current['layer']][current['module']] = {}



def interpolate_features(tensor_list, target_T, prevprev_tensor=None, stage_ratio=0.0):
    """
    GPU 전용 Linear Interpolation (양 끝점 고정)
    tensor_list: list of [B, C, H, W] tensors on GPU
    target_T: 원하는 길이
    """
    if len(tensor_list) == 0:
        raise ValueError("tensor_list is empty.")

    device = tensor_list[0].device
    orig_T = len(tensor_list)
    stacked = torch.stack(tensor_list, dim=0)  # [T, B, C, H, W]
    orig_shape = stacked.shape[1:]             # [B, C, H, W]

    # flatten for interpolation
    flattened = stacked.reshape(orig_T, -1)    # [T, D]

    # original and target positions
    orig_x = torch.linspace(0, 1, orig_T, device=device)
    target_x = torch.linspace(0, 1, target_T, device=device)

    # find indices for interpolation (exclude exact 0 and 1)
    idx = torch.clamp(torch.searchsorted(orig_x, target_x) - 1, 0, orig_T - 2)
    x0 = orig_x[idx]
    x1 = orig_x[idx + 1]
    f0 = flattened[idx]       # [target_T, D]
    f1 = flattened[idx + 1]

    # linear interpolation
    dx = (target_x - x0) / (x1 - x0 + 1e-10)
    interp = f0 + dx.unsqueeze(1) * (f1 - f0)

    # reshape back
    interp_tensor = interp.reshape(target_T, *orig_shape)

    # === enforce exact endpoints ===
    interp_tensor[0]  = stacked[0]
    interp_tensor[-1] = stacked[-1]

    return interp_tensor



## 그냥 인접 step이랑 feature 변화량
def get_interval(prev_features, curr_features, prev_interval, 
                 high_th=0.7, low_th=0.5, 
                 min_interval=3, max_interval=6):
    """
    Adaptive interval scheduling rule.
    Feature 변화량에 따라 다음 coarse interval 길이를 동적으로 조정함.
    """
    if prev_features is None or curr_features is None:
        return prev_interval  # 첫 segment에서는 그대로 유지
    
    #변화량 계산 (attention + MLP)
    diff_attn = torch.mean(torch.abs(curr_features["attn"] - prev_features["attn"])).item()
    diff_mlp  = torch.mean(torch.abs(curr_features["mlp"]  - prev_features["mlp"])).item()
    diff_mean = (diff_attn + diff_mlp) / 2

    # 2️⃣ Adaptive rule
    if diff_mean > high_th:
        new_interval = max(prev_interval - 1, min_interval)
        status = f" High dynamic ({diff_mean:.4f}) -> {new_interval}"
    elif diff_mean < low_th:
        new_interval = min(prev_interval + 1, max_interval)
        status = f" Stable ({diff_mean:.4f}) -> {new_interval}"
    else:
        new_interval = prev_interval
        status = f"Moderate ({diff_mean:.4f}) = {new_interval}"

    print(f"[get_interval] {status}")
    return new_interval


## Curvature로 
def get_interval_by_feature_curv(F_prevprev, F_prev, F_curr, prev_interval,
                                 high_th=1, low_th=0.5,
                                 min_interval=3, max_interval=6):
    """
    Adaptive coarse-step interval scheduling based on feature curvature.
    
    입력:
        F_prevprev: feature at t-2Δ (tensor)
        F_prev:     feature at t-Δ (tensor)
        F_curr:     feature at t   (tensor)
        prev_interval: 이전 coarse interval 길이
    규칙:
        curvature = mean(|(F_t - F_{t-Δ}) - (F_{t-Δ} - F_{t-2Δ})|)
        curvature ↑  → 변화 급격 → interval 감소
        curvature ↓  → 안정적 → interval 증가
    """

    # 초반부
    if F_prevprev is None or F_prev is None or F_curr is None:
        return prev_interval

    # curvature 계산
    curv_attn = torch.mean(torch.abs(
        (F_curr["attn"] - F_prev["attn"]) - (F_prev["attn"] - F_prevprev["attn"])
    )).item()

    curv_mlp = torch.mean(torch.abs(
        (F_curr["mlp"] - F_prev["mlp"]) - (F_prev["mlp"] - F_prevprev["mlp"])
    )).item()

    curvature = (curv_attn)

    # adaptive
    if curvature > high_th:
        new_interval = max(prev_interval - 1, min_interval)
        status = f"⚠️ High curvature ({curvature:.4f}) → Decrease interval → {new_interval}"
    elif curvature < low_th:
        new_interval = min(prev_interval + 1, max_interval)
        status = f"✅ Stable ({curvature:.4f}) → Increase interval → {new_interval}"
    else:
        new_interval = prev_interval
        status = f"~ Moderate ({curvature:.4f}) → Keep interval = {new_interval}"

    print(f"[get_interval_by_feature_curv] {status}")
    return new_interval


def get_interval(curr):
    """
    현재 step(curr)에 따라 고정된 coarse interval 반환
      - 0 ~ 16  → 5
      - 17 ~ 33 → 3
      - 34 ~ 50 → 4
    """
    if curr <= 10:
        new_interval = 5
    elif curr <= 28:
        new_interval = 3
    else:
        new_interval = 4

    return new_interval