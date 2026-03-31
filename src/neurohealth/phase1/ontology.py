from __future__ import annotations

from collections import defaultdict

from .types import KnowledgeChunk, RoutingRule, TriageRule


def build_symptom_condition_graph(
    triage_rules: list[TriageRule],
    knowledge_chunks: list[KnowledgeChunk],
) -> dict:
    topic_to_chunks: dict[str, list[str]] = defaultdict(list)
    for chunk in knowledge_chunks:
        for topic in chunk.medical_topics:
            topic_to_chunks[topic].append(chunk.chunk_id)

    nodes = []
    edges = []

    for rule in triage_rules:
        node_id = f"rule:{rule.rule_id}"
        nodes.append(
            {
                "node_id": node_id,
                "type": "triage_rule",
                "urgency": rule.urgency_level.value,
                "age_group": rule.age_group.value,
                "symptom_pattern": rule.symptom_pattern,
            }
        )

        for symptom in rule.symptom_pattern:
            symptom_node = f"symptom:{symptom.lower().replace(' ', '_')}"
            nodes.append({"node_id": symptom_node, "type": "symptom", "label": symptom})
            edges.append({"from": symptom_node, "to": node_id, "relation": "mapped_to_triage_rule"})

            matched_topics = [
                topic
                for topic in topic_to_chunks.keys()
                if symptom.split(" ")[0].lower() in topic.lower()
            ]
            for topic in matched_topics:
                topic_node = f"topic:{topic}"
                nodes.append({"node_id": topic_node, "type": "medical_topic", "label": topic})
                edges.append({"from": node_id, "to": topic_node, "relation": "supported_by_topic"})

    dedup_nodes = {node["node_id"]: node for node in nodes}
    dedup_edges = {f"{e['from']}->{e['to']}:{e['relation']}": e for e in edges}

    return {
        "nodes": list(dedup_nodes.values()),
        "edges": list(dedup_edges.values()),
        "stats": {
            "node_count": len(dedup_nodes),
            "edge_count": len(dedup_edges),
        },
    }


def load_triage_rules(raw_rules: list[dict]) -> list[TriageRule]:
    return [TriageRule(**row) for row in raw_rules]


def load_routing_rules(raw_rules: list[dict]) -> list[RoutingRule]:
    return [RoutingRule(**row) for row in raw_rules]
