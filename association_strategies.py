import torch


def get_backend_output(model_type, tokens, backend_models, inference_cache):
    if model_type in inference_cache:
        return inference_cache[model_type]

    model = backend_models[model_type]
    with torch.no_grad():
        decision = model.predict_action(tokens)
        probabilities = decision["joint_probabilities"]
        action = int(decision["action"].item())
        if action == probabilities.shape[-1] - 1:
            confidence = float(decision["arrival_probability"].item())
        else:
            confidence = float(decision["move_probabilities"][0, action].item())

    output = {
        # 兼容旧代码：logits字段保留4维log-probabilities。
        "logits": torch.log(probabilities.clamp_min(1e-12)),
        "probabilities": probabilities.squeeze(0),
        "arrival_probability": float(decision["arrival_probability"].item()),
        "move_probabilities": decision["move_probabilities"].squeeze(0),
        "stop_threshold": float(decision["stop_threshold"]),
        "action": action,
        "confidence": confidence,
    }
    inference_cache[model_type] = output
    return output


def _latency_metrics(node, uav_position, tokens, latency_model, tx_power_w, num_actions):
    distance = node.distance_to(uav_position)
    return latency_model.evaluate_step_from_distance(
        tokens=tokens,
        model_type=node.model_type,
        horizontal_distance_m=distance,
        compute_capacity_flops_per_s=node.compute_capacity_flops_per_s,
        tx_power_w=tx_power_w,
        num_actions=num_actions,
    )


def select_nearest(candidates, uav_position):
    return min(candidates, key=lambda node: node.distance_to(uav_position))


def select_random(candidates, rng):
    return candidates[int(rng.integers(0, len(candidates)))]


def select_max_rate(candidates, uav_position, latency_model, tx_power_w):
    best_node = None
    best_rate = -1.0
    for node in candidates:
        channel = latency_model.a2g_channel(
            node.distance_to(uav_position),
            tx_power_w=tx_power_w,
        )
        if channel["uplink_rate_bps"] > best_rate:
            best_rate = channel["uplink_rate_bps"]
            best_node = node
    return best_node


def select_min_latency(
    candidates,
    uav_position,
    tokens,
    latency_model,
    tx_power_w,
    num_actions,
):
    best_node = None
    best_latency = None
    for node in candidates:
        metrics = _latency_metrics(
            node,
            uav_position,
            tokens,
            latency_model,
            tx_power_w,
            num_actions,
        )
        latency = metrics["total_latency_s"]
        if best_latency is None or latency < best_latency:
            best_latency = latency
            best_node = node
    return best_node


def select_best_quality(
    candidates,
    uav_position,
    tokens,
    latency_model,
    tx_power_w,
    num_actions,
    model_quality_scores,
):
    # Primary key: offline validation quality of the hosted model.
    # Tie-break: lower analytical end-to-end latency.
    best_node = None
    best_score = None
    for node in candidates:
        quality = float(model_quality_scores.get(node.model_type, 0.0))
        latency = _latency_metrics(
            node,
            uav_position,
            tokens,
            latency_model,
            tx_power_w,
            num_actions,
        )["total_latency_s"]
        score = (quality, -latency)
        if best_score is None or score > best_score:
            best_score = score
            best_node = node
    return best_node


def select_oracle(
    candidates,
    uav_position,
    tokens,
    backend_models,
    inference_cache,
    oracle_action,
    latency_model,
    tx_power_w,
    num_actions,
):
    # Privileged upper bound: use the oracle action to identify model outputs
    # that are correct, then choose the lowest-latency correct candidate.
    correct = []
    for node in candidates:
        output = get_backend_output(
            node.model_type,
            tokens,
            backend_models,
            inference_cache,
        )
        if output["action"] == oracle_action:
            latency = _latency_metrics(
                node,
                uav_position,
                tokens,
                latency_model,
                tx_power_w,
                num_actions,
            )["total_latency_s"]
            correct.append((latency, node))

    if correct:
        correct.sort(key=lambda item: item[0])
        return correct[0][1]

    # If no model is exactly correct, choose the model assigning the highest
    # probability to the oracle action; use latency as tie-break.
    best_node = None
    best_score = None
    for node in candidates:
        output = get_backend_output(
            node.model_type,
            tokens,
            backend_models,
            inference_cache,
        )
        p = float(output["probabilities"][oracle_action].item())
        latency = _latency_metrics(
            node,
            uav_position,
            tokens,
            latency_model,
            tx_power_w,
            num_actions,
        )["total_latency_s"]
        score = (p, -latency)
        if best_score is None or score > best_score:
            best_score = score
            best_node = node
    return best_node


def select_gbs(
    strategy,
    candidates,
    uav_position,
    tokens,
    backend_models,
    inference_cache,
    rng,
    oracle_action,
    latency_model,
    tx_power_w,
    num_actions,
    model_quality_scores,
):
    if not candidates:
        raise ValueError("No candidate GBS is available.")

    if strategy == "random":
        return select_random(candidates, rng)
    if strategy == "nearest":
        return select_nearest(candidates, uav_position)
    if strategy == "max_rate":
        return select_max_rate(candidates, uav_position, latency_model, tx_power_w)
    if strategy == "min_latency":
        return select_min_latency(
            candidates,
            uav_position,
            tokens,
            latency_model,
            tx_power_w,
            num_actions,
        )
    if strategy == "best_quality":
        return select_best_quality(
            candidates,
            uav_position,
            tokens,
            latency_model,
            tx_power_w,
            num_actions,
            model_quality_scores,
        )
    if strategy == "oracle":
        return select_oracle(
            candidates,
            uav_position,
            tokens,
            backend_models,
            inference_cache,
            oracle_action,
            latency_model,
            tx_power_w,
            num_actions,
        )

    raise ValueError(f"Unknown association strategy: {strategy}")
