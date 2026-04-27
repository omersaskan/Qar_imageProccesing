# Makefile for Meshysiz Product Asset Factory
# Note: Makefile is for convenience. Use direct python commands as canonical fallback.

.PHONY: help install test smoke clean cleanup lint

help:
	@echo "Usage:"
	@echo "  make install   - Install dependencies"
	@echo "  make test      - Run full test suite"
	@echo "  make smoke     - Run end-to-end smoke test"
	@echo "  make lint      - Run basic linting (ruff/flake8 if available)"
	@echo "  make cleanup   - Remove temporary data and cache"

install:
	pip install .
	pip install -e .[dev]

test:
	python -m pytest modules/shared_contracts/tests modules/capture_workflow/tests modules/reconstruction_engine/tests modules/asset_cleanup_pipeline/tests modules/export_pipeline/tests modules/qa_validation/tests modules/asset_registry/tests modules/operations/tests modules/utils/tests modules/tests/test_phase6_integration.py modules/tests/test_smoke_flow.py modules/tests/test_phase7_edge_cases.py modules/tests/test_worker_finalize_flow.py modules/tests/test_worker_import.py tests/test_worker_failure_flow.py tests/test_mask_alignment.py tests/test_sprint1_integration.py tests/test_sprint2_integration.py tests/test_sprint3_integration.py

ci:
	python -m pytest tests/capture_workflow -q
	python -m pytest tests/reconstruction_engine -q
	python -m pytest tests/asset_cleanup_pipeline -q
	python -m pytest tests/export_pipeline -q
	python -m pytest tests/qa_validation -q
	python -m pytest tests/scripts -q
	python -m pytest tests/test_phase5_safe_copy_isolation.py -q
	python -m pytest tests/modules/capture_workflow/test_object_masker_sam2_flag.py -q

smoke:
	python -m pytest modules/tests/test_smoke_flow.py

lint:
	@echo "Running basic checks..."
	python -m compileall modules/

cleanup:
	rm -rf .pytest_cache/
	rm -rf data/
	rm -rf temp_frames/
	rm -f *.log
	rm -f test_output.txt
	rm -f test_final.txt
