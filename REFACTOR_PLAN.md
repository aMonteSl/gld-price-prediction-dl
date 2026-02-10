# Refactor Plan

## Executive Summary
- Split the Streamlit UI script into small UI modules and controllers.
- Keep Streamlit rendering in UI-only modules; move orchestration into controllers.
- Add service wrappers for training, evaluation, forecasting, recommendation, diagnostics.
- Maintain a strict layering rule to avoid circular imports.
- Refactor incrementally with backup protocol per file.
- Add controller/service tests and keep existing tests passing.
- Preserve behavior and UI output while improving modularity.
- Make it easy to add new assets, models, and tabs without touching core layers.

## Inventory (Large Files and Refactor Candidates)
- src/gldpred/app/streamlit_app.py (approx 1,300+ lines)
  - Responsibilities: layout, sidebar config, all tabs, data loading, training and fine-tune orchestration, evaluation metrics, forecasting, recommendations, registry management, compare logic, chart rendering, session state.
  - Pain points: tightly coupled UI and logic, duplicated patterns, hard to test, risk of circular imports.

## Target Architecture (Layered)

### Layers and Rules
- app/ui: Streamlit rendering only, no business logic.
- app/controllers: session state orchestration, calls services, returns view models.
- services: domain workflows without Streamlit; use pure Python types.
- domain/infra: existing packages (data, features, models, training, evaluation, inference, decision, diagnostics, registry, i18n).

### Proposed Tree (Additions Only)
- src/gldpred/app/
  - ui/
    - __init__.py
    - sidebar.py
    - tabs_data.py
    - tabs_train.py
    - tabs_models.py
    - tabs_forecast.py
    - tabs_recommendation.py
    - tabs_evaluation.py
    - tabs_compare.py
    - tabs_tutorial.py
    - components.py
  - controllers/
    - __init__.py
    - training_controller.py
    - evaluation_controller.py
    - forecasting_controller.py
    - recommendation_controller.py
    - models_controller.py
    - compare_controller.py
- src/gldpred/services/
  - __init__.py
  - training_service.py
  - evaluation_service.py
  - forecasting_service.py
  - recommendation_service.py
  - diagnostics_service.py

## Step-by-Step Refactor Plan (Incremental)

### Step 1: Extract UI Helpers and Constants (Low Risk)
- Candidate file: src/gldpred/app/streamlit_app.py
- New files:
  - src/gldpred/app/ui/components.py (diagnostics panel, apply-suggestions, shared helpers)
- Mapping:
  - _show_diagnostics -> ui.components.show_diagnostics
  - _apply_suggestions -> ui.components.apply_suggestions
  - _step_value -> ui.components.step_value
  - _HIDDEN_SIZE_OPTIONS, _BATCH_SIZE_OPTIONS, _LR_OPTIONS -> ui.components.*
- Dependencies:
  - UI components import streamlit, state, plots, glossary.
  - streamlit_app imports ui.components.
- Tests:
  - Import-only sanity check for ui.components.

### Step 2: Split Sidebar and Tabs into UI Modules
- Candidate file: src/gldpred/app/streamlit_app.py
- New files:
  - ui/sidebar.py
  - ui/tabs_data.py
  - ui/tabs_train.py
  - ui/tabs_models.py
  - ui/tabs_forecast.py
  - ui/tabs_recommendation.py
  - ui/tabs_evaluation.py
  - ui/tabs_compare.py
  - ui/tabs_tutorial.py
- Mapping (examples):
  - _sidebar -> ui.sidebar.render_sidebar
  - _tab_data -> ui.tabs_data.render
  - _tab_train -> ui.tabs_train.render
  - _tab_models -> ui.tabs_models.render
  - _tab_forecast -> ui.tabs_forecast.render
  - _tab_recommendation and _show_recommendation_history -> ui.tabs_recommendation.render
  - _tab_evaluation -> ui.tabs_evaluation.render
  - _tab_compare and _render_comparison -> ui.tabs_compare.render
  - _tab_tutorial -> ui.tabs_tutorial.render
- Dependencies:
  - UI modules only call controllers for data, training, evaluation, etc.

### Step 3: Introduce Controllers
- New files:
  - app/controllers/training_controller.py
  - app/controllers/evaluation_controller.py
  - app/controllers/forecasting_controller.py
  - app/controllers/recommendation_controller.py
  - app/controllers/models_controller.py
  - app/controllers/compare_controller.py
- Mapping:
  - Training orchestration (sequence creation, trainer lifecycle) -> training_controller
  - Evaluation metrics -> evaluation_controller
  - Forecasting + fan chart data -> forecasting_controller
  - Recommendation + history -> recommendation_controller
  - Registry actions (rename, delete, primary) -> models_controller
  - Compare orchestration -> compare_controller (merge existing logic)
- Tests:
  - Add unit tests for each controller with mocked services.

### Step 4: Add Service Layer
- New files:
  - services/training_service.py
  - services/evaluation_service.py
  - services/forecasting_service.py
  - services/recommendation_service.py
  - services/diagnostics_service.py
- Responsibilities:
  - Wrap FeatureEngineering, ModelTrainer, ModelEvaluator, TrajectoryPredictor, DecisionEngine, DiagnosticsAnalyzer.
  - No Streamlit imports.
- Tests:
  - Service-level tests using existing fixtures.

### Step 5: Refine Dependencies and Remove Duplications
- Enforce UI -> controller -> service -> domain layering.
- Remove direct domain calls from UI modules.
- Replace repeated error handling with controller-level helpers.

## Refactor Execution Checklist
- Create <filename>.bak.md adjacent to each refactored file with full original content.
- Perform the split/refactor into new files.
- Run import checks, pytest -q, and Streamlit smoke check.
- If any check fails, restore from backup and stop.
- Delete .bak.md only after all checks pass.

## Potential Breakages and Detection
- Circular imports: enforce strict layering; use import checks.
- UI regressions: run Streamlit smoke tests after each step.
- Training/evaluation drift: verify metrics and run pytest -q.
- Session-state regressions: rely on state.py as single source of truth.
- i18n key mismatch: run i18n key audit test.

## Quick Wins
- Extract diagnostics helpers and select-slider options.
- Split sidebar into its own UI module.
- Introduce controllers as pass-through wrappers before logic migration.
- Add an i18n audit test to prevent missing translations.
