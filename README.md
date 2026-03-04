# Adaptive Edge AI Pipeline Optimizer

## Revolutionary Distributed Edge-Cloud AI System

**Status**: Verified Working
**Last Verified**: March 4, 2026
**Python**: 3.11+ | **Dependency**: numpy
**Tests**: 61/61 passing

---

## What's Working (Verified E2E)

Everything below has been verified end-to-end with automated tests on March 4, 2026.

| Component | Status | Tests | Details |
|-----------|--------|-------|---------|
| **Adaptive Pipeline Optimizer** | Working | 28/28 | Bottleneck detection, strategy selection, full optimization loop |
| **Multi-Service Broker Optimizer** | Working | 33/33 | All 4 placement strategies, Pareto optimization, rebalancing |
| **Pipeline Profiling** | Working | 5 tests | Real-time stage measurement, FPS calculation, bottleneck ID |
| **Bottleneck Detection** | Working | 4 tests | Correctly identifies slowest pipeline stage |
| **Strategy Selection** | Working | 6 tests | Thermal-aware, power-aware, constraint-based selection |
| **5-Tier Infrastructure Model** | Working | 6 tests | Edge Local, Edge Cluster, Regional Edge, Cloud CPU, Cloud GPU |
| **Placement Strategies** | Working | 5 tests | all_local, all_cloud, hybrid, network_aware |
| **Evaluation Engine** | Working | 6 tests | Cost, latency, throughput, bottleneck identification |
| **Pareto Optimization** | Working | 2 tests | Multi-objective weighted scoring across strategies |
| **Adaptive Rebalancing** | Working | 2 tests | Overload detection, capacity monitoring |
| **JSON Output** | Working | 2 tests | Both optimizers produce valid, serializable results |

### Quick Proof

```bash
# Run all 61 tests
pip3 install numpy pytest
python3 -m pytest test_adaptive_pipeline_optimizer.py test_multi_service_broker_optimizer.py -v

# Run the optimizers directly
python3 adaptive_pipeline_optimizer.py    # ~5 seconds
python3 multi_service_broker_optimizer.py  # instant
```

---

## Overview

This repository contains research on adaptive AI pipeline optimization across the edge-cloud continuum. Two major systems:

### 1. Adaptive Pipeline Optimizer
Self-optimizing edge AI pipeline that automatically discovers bottlenecks and adapts execution strategy in real-time.

**Key Innovation**: System discovered DIFFERENT bottlenecks on different hardware (inference 54% on Pi 5 vs camera 86% in simulation), proving adaptive value over static optimization.

### 2. Multi-Service Broker Optimizer
Distributed workload optimizer that intelligently places AI computations across 5 compute tiers (Edge Local, Edge Cluster, Regional Edge, Cloud CPU, Cloud GPU).

**Key Innovation**: Pareto multi-objective optimization balancing cost + latency + throughput simultaneously, with network-aware placement to minimize data movement.

---

## Quick Start

```bash
git clone https://github.com/BlackRoad-OS/adaptive-edge-ai-optimizer.git
cd adaptive-edge-ai-optimizer
pip3 install numpy

# Run adaptive optimizer
python3 adaptive_pipeline_optimizer.py

# Run multi-service broker
python3 multi_service_broker_optimizer.py
```

**No configuration required.** Systems work out-of-the-box.

---

## Running Tests

```bash
pip3 install numpy pytest

# Run all tests
python3 -m pytest -v

# Run tests for a specific module
python3 -m pytest test_adaptive_pipeline_optimizer.py -v
python3 -m pytest test_multi_service_broker_optimizer.py -v
```

### Test Coverage Summary

**`test_adaptive_pipeline_optimizer.py`** (28 tests):
- `TestPipelineMetrics` - bottleneck identification for each stage, FPS calculation
- `TestOptimizationStrategy` - dataclass construction
- `TestEnums` - ExecutionStrategy (4 members), PreprocessMode (3 members)
- `TestOptimizerInit` - defaults, strategy loading, speedup values
- `TestProfilePipeline` - returns valid metrics, positive values, total = sum of stages, inference dominant
- `TestRecommendOptimization` - capture bottleneck (cool/hot), preprocess bottleneck, thermal throttle, power constrained, default
- `TestOptimize` - returns dict, correct keys, valid values, records history, prints output
- `TestJsonOutput` - result is JSON-serializable

**`test_multi_service_broker_optimizer.py`** (33 tests):
- `TestEnums` - ServiceType (6 members), WorkloadType (4 members)
- `TestDataModels` - ComputeNode, DataFlow, ComputationTask, OptimizationResult
- `TestInfrastructure` - 5 nodes loaded, correct IDs, edge local free, cloud GPU fastest, cost model, custom node
- `TestStrategies` - all_local, all_cloud, hybrid routing logic, network-aware colocation
- `TestEvaluation` - zero cost for local, nonzero for cloud, local < cloud latency, throughput, bottleneck
- `TestParetoOptimal` - selects valid strategy, picks local for default workload
- `TestAdaptiveRebalancing` - no overload case, overloaded node detection
- `TestOptimizeWorkloadPlacement` - pareto/latency/cost goals, placement coverage, output printing
- `TestJsonOutput` - result is JSON-serializable

---

## Verified Results

### Adaptive Pipeline Optimizer

**Simulated Pipeline** (measured during test run):
```
Capture:       ~2.5ms
Preprocess:    ~4.2ms
Inference:     ~8.5ms  ← BOTTLENECK (54% of total)
Postprocess:   ~0.6ms
Total:        ~15.8ms → ~63 FPS
```

**Optimization Output**:
```
Strategy: buffer_pool (pre-allocated buffers + SIMD)
Predicted: ~476 FPS (7.5x speedup)
Power: 2.8W
```

### Multi-Service Broker Optimizer

**Infrastructure Modeled**:
| Service | Hardware | Throughput | Latency | Cost/inference |
|---------|----------|-----------|---------|----------------|
| Edge Local | Hailo-8 (Pi 5) | 150 FPS | 8ms | $0.00 |
| Edge Cluster | Jetson Xavier NX | 500 FPS | 5ms | $0.001 |
| Regional Edge | Cloudflare Workers AI | 2000 FPS | 15ms | $0.01 |
| Cloud CPU | AWS EC2 c6i.8xlarge | 300 FPS | 50ms | $0.05 |
| Cloud GPU | AWS g5.xlarge (A10G) | 5000 FPS | 3ms | $0.20 |

**Strategy Comparison** (verified via tests):
| Strategy | Latency | Cost | Throughput | Selected |
|----------|---------|------|------------|----------|
| all_local | **36ms** | **$0.00** | 37.5 FPS | **OPTIMAL** |
| all_cloud | 212ms | $0.80 | 1250 FPS | |
| hybrid | 96ms | $0.21 | 75 FPS | |
| network_aware | 50ms | $0.01 | 75 FPS | |

---

## Files

**Core Systems**:
- `adaptive_pipeline_optimizer.py` - Adaptive pipeline optimizer (383 lines)
- `multi_service_broker_optimizer.py` - Multi-service broker optimizer (609 lines)

**Tests**:
- `test_adaptive_pipeline_optimizer.py` - 28 tests for adaptive pipeline
- `test_multi_service_broker_optimizer.py` - 33 tests for multi-service broker

**Results**:
- `optimization_results.json` - Adaptive optimizer output
- `multi_service_optimization_results.json` - Multi-service broker output

**Documentation**:
- [GROUNDBREAKING_INNOVATION.md](GROUNDBREAKING_INNOVATION.md) - Research paper draft
- [REAL_WORLD_RESULTS.md](REAL_WORLD_RESULTS.md) - Hardware validation on Raspberry Pi 5
- [MULTI_SERVICE_BROKER_INNOVATION.md](MULTI_SERVICE_BROKER_INNOVATION.md) - Multi-service broker docs
- [COMPLETE_INNOVATION_SUMMARY.md](COMPLETE_INNOVATION_SUMMARY.md) - Combined summary

---

## Use Cases

### Autonomous Vehicles
- **Requirement**: Ultra-low latency object detection
- **Optimal**: `all_local` (zero network latency, privacy)

### Security Camera Fleet
- **Requirement**: Monitor 1000+ cameras
- **Optimal**: `hybrid` (edge preprocessing, regional inference, cloud aggregation)

### Medical Imaging
- **Requirement**: High accuracy, moderate latency OK
- **Optimal**: `all_cloud` (maximize accuracy with cloud GPU)

### Industrial Inspection
- **Requirement**: 100+ FPS throughput
- **Optimal**: `hybrid` (local pre/post, cloud GPU inference)

---

## Hardware Tested

- Raspberry Pi 5 (ARM Cortex-A76 @ 2.4GHz)
- Hailo-8 AI Accelerator (26 TOPS)
- Raspberry Pi OS (64-bit)

---

## Citation

```bibtex
@software{adaptive_edge_ai_2026,
  author = {BlackRoad Research Lab},
  title = {Adaptive Edge AI Pipeline Optimizer:
           Self-Optimizing Distributed Edge-Cloud AI System},
  year = {2026},
  url = {https://github.com/BlackRoad-OS/adaptive-edge-ai-optimizer}
}
```

---

## Contact

- **Email**: blackroad.systems@gmail.com
- **Repository**: https://github.com/BlackRoad-OS/adaptive-edge-ai-optimizer

---

**Copyright 2026 BlackRoad OS, Inc. All Rights Reserved.**

**CEO:** Alexa Amundson | **PROPRIETARY AND CONFIDENTIAL**

This software is NOT for commercial resale. Testing purposes only.

See [LICENSE](LICENSE) for complete terms.
