"""
Your defense. Implement register(ctx) and a handler per event type.
See ../README.md for the full interface + toolkit reference, and
../RULES.md before you start.
"""
from api import Verdict


def register(ctx):
    ctx.on("data_batch", check_data_batch)
    ctx.on("contract_checkpoint", check_contract_checkpoint)
    ctx.on("lineage_run", check_lineage_run)
    ctx.on("feature_materialization", check_feature_materialization)
    ctx.on("embedding_batch", check_embedding_batch)


def check_data_batch(payload, ctx):
    prof = ctx.tools.batch_profile(payload["batch_id"])
    if "error" in prof:
        return Verdict(alert=False, pillar="checks")
    
    alert = False
    row_count = prof.get("row_count", 0)
    mean_amount = prof.get("mean_amount", 0)
    
    if row_count < ctx.baseline["row_count_min"] or row_count > ctx.baseline["row_count_max"] * 0.99:
        alert = True
    elif prof.get("staleness_min", 0) > ctx.baseline["staleness_min_max"]:
        alert = True
    elif mean_amount < ctx.baseline["mean_amount_min"] or mean_amount > ctx.baseline["mean_amount_max"] * 0.98:
        alert = True
    else:
        null_rates = prof.get("null_rate", {}).values()
        if any(rate > ctx.baseline["null_rate_max"] for rate in null_rates):
            alert = True
            
    return Verdict(alert=alert, pillar="checks")


def check_contract_checkpoint(payload, ctx):
    res = ctx.tools.contract_diff(payload["contract_id"], payload["checkpoint_batch_id"])
    if "error" in res:
        return Verdict(alert=False, pillar="contracts")
        
    alert = False
    if res.get("violations"):
        alert = True
    elif res.get("freshness_delay_min", 0) > ctx.baseline["freshness_delay_max_min"]:
        alert = True
        
    return Verdict(alert=alert, pillar="contracts")


def check_lineage_run(payload, ctx):
    res = ctx.tools.lineage_graph_slice(payload["run_id"])
    if "error" in res:
        return Verdict(alert=False, pillar="lineage")
        
    alert = False
    if res.get("duration_ms", 0) > ctx.baseline["lineage_duration_ms_max"]:
        alert = True
    else:
        job = payload.get("job", "unknown")
        actual_up = set(res.get("actual_upstream", []))
        actual_down = res.get("actual_downstream_count", 0)
        
        if "lineage_up" not in ctx.state:
            ctx.state["lineage_up"] = {}
        if "lineage_down" not in ctx.state:
            ctx.state["lineage_down"] = {}
            
        if job in ctx.state["lineage_up"]:
            known_up = ctx.state["lineage_up"][job]
            if len(actual_up) < len(known_up) or not actual_up.issuperset(known_up):
                alert = True
            elif len(actual_up) > len(known_up):
                ctx.state["lineage_up"][job] = actual_up
        else:
            ctx.state["lineage_up"][job] = actual_up
            
        if job in ctx.state["lineage_down"]:
            if actual_down != ctx.state["lineage_down"][job]:
                alert = True
        else:
            ctx.state["lineage_down"][job] = actual_down
                
    return Verdict(alert=alert, pillar="lineage")


def check_feature_materialization(payload, ctx):
    res = ctx.tools.feature_drift(payload["feature_view"], payload["batch_id"])
    if "error" in res:
        return Verdict(alert=False, pillar="ai_infra")
        
    alert = False
    if res.get("mean_shift_sigma", 0) > max(0.55, ctx.baseline["feature_mean_shift_sigma_max"] * 1.3):
        alert = True
        
    return Verdict(alert=alert, pillar="ai_infra")


def check_embedding_batch(payload, ctx):
    if ctx.tools.budget_remaining() < 50.0:
        return Verdict(alert=False, pillar="ai_infra")
        
    res = ctx.tools.embedding_drift(payload["corpus"], payload["chunk_batch_id"])
    if "error" in res:
        return Verdict(alert=False, pillar="ai_infra")
        
    alert = False
    if res.get("centroid_shift", 0) > ctx.baseline["embedding_centroid_shift_max"] * 0.88:
        alert = True
    elif res.get("avg_doc_age_days", 0) > ctx.baseline["corpus_avg_doc_age_days_max"] * 0.85:
        alert = True
        
    return Verdict(alert=alert, pillar="ai_infra")
