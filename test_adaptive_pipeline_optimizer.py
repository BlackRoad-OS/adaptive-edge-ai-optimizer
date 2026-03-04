#!/usr/bin/env python3
"""
Tests for Adaptive Edge AI Pipeline Optimizer

Verifies:
- Data model construction and behavior
- Pipeline profiling produces valid metrics
- Bottleneck identification logic
- Optimization strategy selection under various conditions
- Full optimize() pipeline end-to-end
- JSON output generation
"""

import json
import os
import pytest
import numpy as np

from adaptive_pipeline_optimizer import (
    ExecutionStrategy,
    PreprocessMode,
    PipelineMetrics,
    OptimizationStrategy,
    AdaptivePipelineOptimizer,
)


# ---------------------------------------------------------------------------
# Data model tests
# ---------------------------------------------------------------------------

class TestPipelineMetrics:
    """Tests for the PipelineMetrics dataclass."""

    def test_bottleneck_capture(self):
        m = PipelineMetrics(
            capture_ms=100, preprocess_ms=5, inference_ms=8,
            postprocess_ms=1, total_ms=114, fps=8.8,
            thermal_temp=45, power_watts=2.0,
        )
        assert m.bottleneck() == "capture"

    def test_bottleneck_preprocess(self):
        m = PipelineMetrics(
            capture_ms=1, preprocess_ms=50, inference_ms=8,
            postprocess_ms=1, total_ms=60, fps=16.7,
            thermal_temp=45, power_watts=2.0,
        )
        assert m.bottleneck() == "preprocess"

    def test_bottleneck_inference(self):
        m = PipelineMetrics(
            capture_ms=3, preprocess_ms=2, inference_ms=80,
            postprocess_ms=1, total_ms=86, fps=11.6,
            thermal_temp=55, power_watts=2.5,
        )
        assert m.bottleneck() == "inference"

    def test_bottleneck_postprocess(self):
        m = PipelineMetrics(
            capture_ms=1, preprocess_ms=1, inference_ms=1,
            postprocess_ms=50, total_ms=53, fps=18.9,
            thermal_temp=40, power_watts=1.5,
        )
        assert m.bottleneck() == "postprocess"

    def test_fps_calculation(self):
        m = PipelineMetrics(
            capture_ms=5, preprocess_ms=5, inference_ms=5,
            postprocess_ms=5, total_ms=20, fps=1000 / 20,
            thermal_temp=50, power_watts=2.0,
        )
        assert m.fps == pytest.approx(50.0)


class TestOptimizationStrategy:
    """Tests for the OptimizationStrategy dataclass."""

    def test_construction(self):
        s = OptimizationStrategy(
            name="test",
            execution=ExecutionStrategy.BUFFER_POOL,
            preprocess=PreprocessMode.CPU_SIMD,
            predicted_speedup=5.0,
            predicted_power=2.0,
            description="test strategy",
        )
        assert s.name == "test"
        assert s.execution == ExecutionStrategy.BUFFER_POOL
        assert s.predicted_speedup == 5.0


class TestEnums:
    """Verify enum members exist."""

    def test_execution_strategies(self):
        assert len(ExecutionStrategy) == 4
        assert ExecutionStrategy.CAMERA_DIRECT.value == "camera_direct"
        assert ExecutionStrategy.BUFFER_POOL.value == "buffer_pool"
        assert ExecutionStrategy.ZERO_COPY.value == "zero_copy"
        assert ExecutionStrategy.ASYNC_PIPELINE.value == "async_pipeline"

    def test_preprocess_modes(self):
        assert len(PreprocessMode) == 3
        assert PreprocessMode.CPU_NAIVE.value == "cpu_naive"
        assert PreprocessMode.CPU_SIMD.value == "cpu_simd"
        assert PreprocessMode.NPU_OFFLOAD.value == "npu_offload"


# ---------------------------------------------------------------------------
# Optimizer initialization tests
# ---------------------------------------------------------------------------

class TestOptimizerInit:
    """Tests for AdaptivePipelineOptimizer construction."""

    def test_default_init(self):
        opt = AdaptivePipelineOptimizer()
        assert opt.current_strategy == ExecutionStrategy.CAMERA_DIRECT
        assert opt.thermal_threshold == 65.0
        assert opt.power_budget == 5.0
        assert len(opt.history) == 0

    def test_strategies_loaded(self):
        opt = AdaptivePipelineOptimizer()
        assert "baseline" in opt.strategies
        assert "buffer_pool" in opt.strategies
        assert "zero_copy" in opt.strategies
        assert "async_thermal" in opt.strategies

    def test_strategy_speedups(self):
        opt = AdaptivePipelineOptimizer()
        assert opt.strategies["baseline"].predicted_speedup == 1.0
        assert opt.strategies["buffer_pool"].predicted_speedup == 7.5
        assert opt.strategies["zero_copy"].predicted_speedup == 12.0
        assert opt.strategies["async_thermal"].predicted_speedup == 5.0


# ---------------------------------------------------------------------------
# Pipeline profiling tests
# ---------------------------------------------------------------------------

class TestProfilePipeline:
    """Tests for profile_pipeline()."""

    def test_returns_metrics(self):
        opt = AdaptivePipelineOptimizer()
        metrics = opt.profile_pipeline(num_samples=5)
        assert isinstance(metrics, PipelineMetrics)

    def test_metrics_positive(self):
        opt = AdaptivePipelineOptimizer()
        metrics = opt.profile_pipeline(num_samples=5)
        assert metrics.capture_ms > 0
        assert metrics.preprocess_ms > 0
        assert metrics.inference_ms > 0
        assert metrics.postprocess_ms > 0
        assert metrics.total_ms > 0
        assert metrics.fps > 0

    def test_total_is_sum_of_stages(self):
        opt = AdaptivePipelineOptimizer()
        m = opt.profile_pipeline(num_samples=5)
        expected = m.capture_ms + m.preprocess_ms + m.inference_ms + m.postprocess_ms
        assert m.total_ms == pytest.approx(expected, rel=1e-3)

    def test_fps_matches_total(self):
        opt = AdaptivePipelineOptimizer()
        m = opt.profile_pipeline(num_samples=5)
        assert m.fps == pytest.approx(1000.0 / m.total_ms, rel=1e-3)

    def test_inference_is_dominant(self):
        """Inference has an 8ms sleep so it should be the bottleneck."""
        opt = AdaptivePipelineOptimizer()
        m = opt.profile_pipeline(num_samples=10)
        assert m.bottleneck() == "inference"
        assert m.inference_ms >= 7.5  # ~8ms sleep


# ---------------------------------------------------------------------------
# Recommendation logic tests
# ---------------------------------------------------------------------------

class TestRecommendOptimization:
    """Tests for recommend_optimization()."""

    def test_capture_bottleneck_cool(self):
        """Large capture bottleneck + cool temp -> zero_copy."""
        opt = AdaptivePipelineOptimizer()
        m = PipelineMetrics(
            capture_ms=80, preprocess_ms=5, inference_ms=8,
            postprocess_ms=1, total_ms=94, fps=10.6,
            thermal_temp=50.0, power_watts=2.0,
        )
        rec = opt.recommend_optimization(m)
        assert rec.name == "zero_copy"

    def test_capture_bottleneck_hot(self):
        """Large capture bottleneck + hot temp -> buffer_pool."""
        opt = AdaptivePipelineOptimizer()
        m = PipelineMetrics(
            capture_ms=80, preprocess_ms=5, inference_ms=8,
            postprocess_ms=1, total_ms=94, fps=10.6,
            thermal_temp=70.0, power_watts=2.0,
        )
        rec = opt.recommend_optimization(m)
        assert rec.name == "buffer_pool"

    def test_preprocess_bottleneck(self):
        """Large preprocess bottleneck -> zero_copy."""
        opt = AdaptivePipelineOptimizer()
        m = PipelineMetrics(
            capture_ms=3, preprocess_ms=50, inference_ms=8,
            postprocess_ms=1, total_ms=62, fps=16.1,
            thermal_temp=50.0, power_watts=2.0,
        )
        rec = opt.recommend_optimization(m)
        assert rec.name == "zero_copy"

    def test_thermal_throttle(self):
        """High temp (no other bottleneck triggers) -> async_thermal."""
        opt = AdaptivePipelineOptimizer()
        m = PipelineMetrics(
            capture_ms=3, preprocess_ms=3, inference_ms=8,
            postprocess_ms=1, total_ms=15, fps=66.7,
            thermal_temp=70.0, power_watts=2.0,
        )
        rec = opt.recommend_optimization(m)
        assert rec.name == "async_thermal"

    def test_power_constrained(self):
        """Power over budget -> async_thermal."""
        opt = AdaptivePipelineOptimizer()
        m = PipelineMetrics(
            capture_ms=3, preprocess_ms=3, inference_ms=8,
            postprocess_ms=1, total_ms=15, fps=66.7,
            thermal_temp=50.0, power_watts=6.0,
        )
        rec = opt.recommend_optimization(m)
        assert rec.name == "async_thermal"

    def test_default_recommendation(self):
        """No special conditions -> buffer_pool."""
        opt = AdaptivePipelineOptimizer()
        m = PipelineMetrics(
            capture_ms=3, preprocess_ms=3, inference_ms=8,
            postprocess_ms=1, total_ms=15, fps=66.7,
            thermal_temp=50.0, power_watts=2.0,
        )
        rec = opt.recommend_optimization(m)
        assert rec.name == "buffer_pool"


# ---------------------------------------------------------------------------
# Full optimize() pipeline test
# ---------------------------------------------------------------------------

class TestOptimize:
    """End-to-end test of optimize()."""

    def test_optimize_returns_dict(self, capsys):
        opt = AdaptivePipelineOptimizer()
        result = opt.optimize()
        assert isinstance(result, dict)

    def test_optimize_result_keys(self, capsys):
        opt = AdaptivePipelineOptimizer()
        result = opt.optimize()
        expected_keys = {
            "baseline_fps", "optimized_fps", "speedup",
            "bottleneck", "strategy", "implementation",
        }
        assert expected_keys == set(result.keys())

    def test_optimize_values_valid(self, capsys):
        opt = AdaptivePipelineOptimizer()
        result = opt.optimize()
        assert result["baseline_fps"] > 0
        assert result["optimized_fps"] > result["baseline_fps"]
        assert result["speedup"] > 1.0
        assert result["bottleneck"] in ("capture", "preprocess", "inference", "postprocess")
        assert result["strategy"] in ("baseline", "buffer_pool", "zero_copy", "async_thermal")

    def test_optimize_records_history(self, capsys):
        opt = AdaptivePipelineOptimizer()
        opt.optimize()
        assert len(opt.history) == 1

    def test_optimize_prints_output(self, capsys):
        opt = AdaptivePipelineOptimizer()
        opt.optimize()
        captured = capsys.readouterr()
        assert "ADAPTIVE EDGE AI PIPELINE OPTIMIZER" in captured.out
        assert "Profiling current pipeline" in captured.out
        assert "BOTTLENECK" in captured.out
        assert "STRATEGY" in captured.out


# ---------------------------------------------------------------------------
# JSON serialization test
# ---------------------------------------------------------------------------

class TestJsonOutput:
    """Verify results are JSON-serializable."""

    def test_result_json_serializable(self, capsys):
        opt = AdaptivePipelineOptimizer()
        result = opt.optimize()
        serialized = json.dumps(result, indent=2)
        loaded = json.loads(serialized)
        assert loaded["bottleneck"] == result["bottleneck"]
        assert loaded["speedup"] == result["speedup"]
