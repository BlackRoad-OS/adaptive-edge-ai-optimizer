#!/usr/bin/env python3
"""
Tests for Multi-Service Broker Computation Optimizer

Verifies:
- Data model construction
- Infrastructure initialization (5 compute tiers)
- Strategy generation (all_local, all_cloud, hybrid, network_aware)
- Placement evaluation metrics
- Pareto optimization selection
- Adaptive rebalancing
- Full end-to-end optimize_workload_placement()
- JSON output generation
"""

import json
import pytest
import numpy as np

from multi_service_broker_optimizer import (
    ServiceType,
    WorkloadType,
    ComputeNode,
    DataFlow,
    ComputationTask,
    OptimizationResult,
    MultiServiceBrokerOptimizer,
)


# ---------------------------------------------------------------------------
# Helper: standard task set used across tests
# ---------------------------------------------------------------------------

def make_tasks():
    """Create the standard 4-task video analytics pipeline."""
    return [
        ComputationTask(
            task_id="preprocess_1",
            workload_type=WorkloadType.PREPROCESSING,
            input_size_mb=5.0, output_size_mb=2.0,
            compute_complexity=10.0,
            max_latency_ms=10.0, min_accuracy=0.99, max_cost_cents=0.001,
        ),
        ComputationTask(
            task_id="inference_1",
            workload_type=WorkloadType.INFERENCE,
            input_size_mb=2.0, output_size_mb=0.1,
            compute_complexity=500.0,
            max_latency_ms=50.0, min_accuracy=0.95, max_cost_cents=0.10,
        ),
        ComputationTask(
            task_id="postprocess_1",
            workload_type=WorkloadType.POSTPROCESSING,
            input_size_mb=0.1, output_size_mb=0.01,
            compute_complexity=5.0,
            max_latency_ms=5.0, min_accuracy=1.0, max_cost_cents=0.001,
        ),
        ComputationTask(
            task_id="aggregation_1",
            workload_type=WorkloadType.AGGREGATION,
            input_size_mb=0.5, output_size_mb=0.01,
            compute_complexity=20.0,
            max_latency_ms=100.0, min_accuracy=1.0, max_cost_cents=0.01,
        ),
    ]


# ---------------------------------------------------------------------------
# Enum tests
# ---------------------------------------------------------------------------

class TestEnums:
    def test_service_types(self):
        assert len(ServiceType) == 6  # includes HYBRID
        assert ServiceType.EDGE_LOCAL.value == "edge_local"
        assert ServiceType.CLOUD_GPU.value == "cloud_gpu"

    def test_workload_types(self):
        assert len(WorkloadType) == 4
        assert WorkloadType.INFERENCE.value == "inference"


# ---------------------------------------------------------------------------
# Data model tests
# ---------------------------------------------------------------------------

class TestDataModels:
    def test_compute_node(self):
        node = ComputeNode(
            node_id="test", service_type=ServiceType.EDGE_LOCAL,
            hardware="test-hw", throughput_fps=100, latency_ms=5,
            cost_per_inference=0.0, current_load=0.0, max_concurrent=1,
            available_memory_mb=1024, network_latency_ms=1.0,
            bandwidth_mbps=1000, uptime_sla=0.99, failure_rate=0.01,
        )
        assert node.node_id == "test"
        assert node.throughput_fps == 100

    def test_data_flow(self):
        df = DataFlow(source="a", destination="b", data_size_mb=1.0,
                      priority=5, max_latency_ms=10.0)
        assert df.source == "a"

    def test_computation_task(self):
        t = ComputationTask(
            task_id="t1", workload_type=WorkloadType.INFERENCE,
            input_size_mb=2.0, output_size_mb=0.1,
            compute_complexity=100.0, max_latency_ms=50.0,
            min_accuracy=0.95, max_cost_cents=0.10,
        )
        assert t.compute_complexity == 100.0

    def test_optimization_result(self):
        r = OptimizationResult(
            placement={"t1": "n1"}, total_latency_ms=10.0,
            total_cost_cents=0.01, throughput_fps=100.0,
            data_flows=[], bottleneck_node="n1",
            optimization_strategy="test",
        )
        assert r.throughput_fps == 100.0


# ---------------------------------------------------------------------------
# Infrastructure initialization tests
# ---------------------------------------------------------------------------

class TestInfrastructure:
    def test_five_nodes_loaded(self):
        opt = MultiServiceBrokerOptimizer()
        assert len(opt.compute_nodes) == 5

    def test_node_ids(self):
        opt = MultiServiceBrokerOptimizer()
        expected = {"pi5-hailo-1", "jetson-cluster", "cloudflare-edge",
                    "aws-cpu-east1", "aws-gpu-east1"}
        assert set(opt.compute_nodes.keys()) == expected

    def test_edge_local_is_free(self):
        opt = MultiServiceBrokerOptimizer()
        assert opt.compute_nodes["pi5-hailo-1"].cost_per_inference == 0.0

    def test_cloud_gpu_highest_throughput(self):
        opt = MultiServiceBrokerOptimizer()
        gpu = opt.compute_nodes["aws-gpu-east1"]
        for nid, node in opt.compute_nodes.items():
            if nid != "aws-gpu-east1":
                assert gpu.throughput_fps >= node.throughput_fps

    def test_cost_model(self):
        opt = MultiServiceBrokerOptimizer()
        assert opt.cost_model[ServiceType.EDGE_LOCAL] == 0.0
        assert opt.cost_model[ServiceType.CLOUD_GPU] == 0.20

    def test_add_custom_node(self):
        opt = MultiServiceBrokerOptimizer()
        custom = ComputeNode(
            node_id="custom-1", service_type=ServiceType.EDGE_LOCAL,
            hardware="custom", throughput_fps=200, latency_ms=4,
            cost_per_inference=0.0, current_load=0.0, max_concurrent=2,
            available_memory_mb=2048, network_latency_ms=1.0,
            bandwidth_mbps=1000, uptime_sla=0.99, failure_rate=0.01,
        )
        opt.add_compute_node(custom)
        assert "custom-1" in opt.compute_nodes
        assert len(opt.compute_nodes) == 6


# ---------------------------------------------------------------------------
# Strategy generation tests
# ---------------------------------------------------------------------------

class TestStrategies:
    def test_all_local(self):
        opt = MultiServiceBrokerOptimizer()
        tasks = make_tasks()
        placement = opt._all_local_strategy(tasks)
        assert all(v == "pi5-hailo-1" for v in placement.values())
        assert len(placement) == 4

    def test_all_cloud(self):
        opt = MultiServiceBrokerOptimizer()
        tasks = make_tasks()
        placement = opt._all_cloud_strategy(tasks)
        assert all(v == "aws-gpu-east1" for v in placement.values())
        assert len(placement) == 4

    def test_hybrid_latency_sensitive_goes_local(self):
        opt = MultiServiceBrokerOptimizer()
        tasks = make_tasks()
        placement = opt._hybrid_strategy(tasks)
        # preprocess_1 has max_latency_ms=10 < 20 → local
        assert placement["preprocess_1"] == "pi5-hailo-1"

    def test_hybrid_compute_heavy_goes_cloud(self):
        opt = MultiServiceBrokerOptimizer()
        tasks = make_tasks()
        placement = opt._hybrid_strategy(tasks)
        # inference_1 has compute_complexity=500 > 100 → cloud GPU
        assert placement["inference_1"] == "aws-gpu-east1"

    def test_network_aware_colocates_by_type(self):
        opt = MultiServiceBrokerOptimizer()
        tasks = make_tasks()
        placement = opt._network_aware_strategy(tasks)
        # Preprocessing and postprocessing should be on local edge
        assert placement["preprocess_1"] == "pi5-hailo-1"
        assert placement["postprocess_1"] == "pi5-hailo-1"
        # Inference goes to jetson cluster
        assert placement["inference_1"] == "jetson-cluster"
        # Aggregation goes to cloudflare
        assert placement["aggregation_1"] == "cloudflare-edge"


# ---------------------------------------------------------------------------
# Evaluation tests
# ---------------------------------------------------------------------------

class TestEvaluation:
    def test_all_local_zero_cost(self):
        opt = MultiServiceBrokerOptimizer()
        tasks = make_tasks()
        placement = opt._all_local_strategy(tasks)
        result = opt._evaluate_placement(tasks, placement)
        assert result.total_cost_cents == 0.0

    def test_all_cloud_nonzero_cost(self):
        opt = MultiServiceBrokerOptimizer()
        tasks = make_tasks()
        placement = opt._all_cloud_strategy(tasks)
        result = opt._evaluate_placement(tasks, placement)
        assert result.total_cost_cents > 0

    def test_all_local_lower_latency_than_cloud(self):
        opt = MultiServiceBrokerOptimizer()
        tasks = make_tasks()
        local_result = opt._evaluate_placement(tasks, opt._all_local_strategy(tasks))
        cloud_result = opt._evaluate_placement(tasks, opt._all_cloud_strategy(tasks))
        assert local_result.total_latency_ms < cloud_result.total_latency_ms

    def test_cloud_higher_throughput_per_node(self):
        opt = MultiServiceBrokerOptimizer()
        gpu_tput = opt.compute_nodes["aws-gpu-east1"].throughput_fps
        local_tput = opt.compute_nodes["pi5-hailo-1"].throughput_fps
        assert gpu_tput > local_tput

    def test_bottleneck_node_populated(self):
        opt = MultiServiceBrokerOptimizer()
        tasks = make_tasks()
        result = opt._evaluate_placement(tasks, opt._all_local_strategy(tasks))
        assert result.bottleneck_node == "pi5-hailo-1"

    def test_throughput_positive(self):
        opt = MultiServiceBrokerOptimizer()
        tasks = make_tasks()
        result = opt._evaluate_placement(tasks, opt._all_local_strategy(tasks))
        assert result.throughput_fps > 0


# ---------------------------------------------------------------------------
# Pareto optimization tests
# ---------------------------------------------------------------------------

class TestParetoOptimal:
    def test_pareto_selects_balanced(self):
        opt = MultiServiceBrokerOptimizer()
        tasks = make_tasks()

        strategies = [
            ("all_local", opt._evaluate_placement(tasks, opt._all_local_strategy(tasks))),
            ("all_cloud", opt._evaluate_placement(tasks, opt._all_cloud_strategy(tasks))),
            ("hybrid", opt._evaluate_placement(tasks, opt._hybrid_strategy(tasks))),
            ("network_aware", opt._evaluate_placement(tasks, opt._network_aware_strategy(tasks))),
        ]

        name, result = opt._pareto_optimal(strategies)
        assert name in ("all_local", "all_cloud", "hybrid", "network_aware")
        assert isinstance(result, OptimizationResult)

    def test_pareto_with_default_tasks_picks_local(self):
        """With default tasks, all_local should win (zero cost, low latency)."""
        opt = MultiServiceBrokerOptimizer()
        tasks = make_tasks()

        strategies = [
            ("all_local", opt._evaluate_placement(tasks, opt._all_local_strategy(tasks))),
            ("all_cloud", opt._evaluate_placement(tasks, opt._all_cloud_strategy(tasks))),
            ("hybrid", opt._evaluate_placement(tasks, opt._hybrid_strategy(tasks))),
            ("network_aware", opt._evaluate_placement(tasks, opt._network_aware_strategy(tasks))),
        ]

        name, _ = opt._pareto_optimal(strategies)
        assert name == "all_local"


# ---------------------------------------------------------------------------
# Adaptive rebalancing tests
# ---------------------------------------------------------------------------

class TestAdaptiveRebalancing:
    def test_no_overload(self, capsys):
        opt = MultiServiceBrokerOptimizer()
        tasks = make_tasks()
        placement = opt._all_local_strategy(tasks)
        result = opt._evaluate_placement(tasks, placement)
        rebalanced = opt.adaptive_rebalancing(result)
        captured = capsys.readouterr()
        assert "All nodes operating within capacity" in captured.out
        assert rebalanced is result

    def test_overloaded_node_detected(self, capsys):
        opt = MultiServiceBrokerOptimizer()
        # Simulate overload
        opt.compute_nodes["pi5-hailo-1"].current_load = 0.95
        tasks = make_tasks()
        result = opt._evaluate_placement(tasks, opt._all_local_strategy(tasks))
        opt.adaptive_rebalancing(result)
        captured = capsys.readouterr()
        assert "overloaded" in captured.out


# ---------------------------------------------------------------------------
# Full end-to-end test
# ---------------------------------------------------------------------------

class TestOptimizeWorkloadPlacement:
    def test_pareto_goal(self, capsys):
        opt = MultiServiceBrokerOptimizer()
        tasks = make_tasks()
        result = opt.optimize_workload_placement(tasks, optimization_goal="pareto")
        assert isinstance(result, OptimizationResult)
        assert result.total_latency_ms > 0
        assert result.throughput_fps > 0

    def test_latency_goal(self, capsys):
        opt = MultiServiceBrokerOptimizer()
        tasks = make_tasks()
        result = opt.optimize_workload_placement(tasks, optimization_goal="latency")
        assert isinstance(result, OptimizationResult)

    def test_cost_goal(self, capsys):
        opt = MultiServiceBrokerOptimizer()
        tasks = make_tasks()
        result = opt.optimize_workload_placement(tasks, optimization_goal="cost")
        assert isinstance(result, OptimizationResult)
        # Cost goal should pick cheapest option
        assert result.total_cost_cents == 0.0

    def test_placement_covers_all_tasks(self, capsys):
        opt = MultiServiceBrokerOptimizer()
        tasks = make_tasks()
        result = opt.optimize_workload_placement(tasks)
        assert set(result.placement.keys()) == {t.task_id for t in tasks}

    def test_prints_summary(self, capsys):
        opt = MultiServiceBrokerOptimizer()
        tasks = make_tasks()
        opt.optimize_workload_placement(tasks)
        captured = capsys.readouterr()
        assert "MULTI-SERVICE BROKER" in captured.out
        assert "SELECTED" in captured.out


# ---------------------------------------------------------------------------
# JSON serialization test
# ---------------------------------------------------------------------------

class TestJsonOutput:
    def test_result_json_serializable(self, capsys):
        opt = MultiServiceBrokerOptimizer()
        tasks = make_tasks()
        result = opt.optimize_workload_placement(tasks)
        output = {
            "placement": result.placement,
            "total_latency_ms": result.total_latency_ms,
            "total_cost_cents": result.total_cost_cents,
            "throughput_fps": result.throughput_fps,
            "bottleneck_node": result.bottleneck_node,
            "strategy": result.optimization_strategy,
        }
        serialized = json.dumps(output, indent=2)
        loaded = json.loads(serialized)
        assert loaded["total_latency_ms"] == result.total_latency_ms
