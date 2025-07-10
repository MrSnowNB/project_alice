---
title: Project Alice v3.0 - Resilience Enhancement Plan
version: 3.0
date: July 10, 2025
author: Grok
objective: |
  To provide a comprehensive, AI-first architectural plan for fixing runtime issues in Project Alice, enhancing resilience, and implementing self-improvement features. This document serves as a guide for human implementation using coding skills and Gemini Code Assist in VSCode, ensuring modularity and minimal disruption to existing code.
core_philosophy:
  - Separation of Concerns: Deterministic Python code handles logic and state; LLM focuses on planning and tool calls.
  - Resilience over Perfection: Build in circuit breakers, error classification, and escalation to handle failures gracefully.
  - Modularity and Runtime Extensibility: Enable dynamic tool creation and sub-agent isolation to prevent monolithic failures.
  - AI First: Prioritize LLM-driven planning with human oversight via VSCode integration for code reviews and assists.
prerequisites:
  - Review existing MD files: Project Alice v1.0, v2.0, v2.5.
  - Ensure LM Studio is running with Devstral-Small-2505 GGUF loaded.
  - Memory Sub-Agent must be active (run memory_service.py).
  - VSCode with Gemini Code Assist extension for refactoring suggestions.
testing_goals:
  - Validate against weather query failure loop.
  - Ensure VRAM < 12GB during operations.
  - Run evaluation harness post-implementation.
---

### **Project Alice v3.0: Resilience Enhancement Plan**

#### **1. Executive Summary**
Project Alice v2.5 is a modular AI agent with strong foundations in LangGraph workflows, sub-agents, and dynamic tool generation. However, runtime issues like infinite replanning loops, tool failures (e.g., HTTP 401/403/429 errors), and state bloat prevent reliable execution. This plan addresses these by introducing circuit breakers, enhanced error handling, and forced extensibility triggers, while aligning with AI-first principles. Implementation will leverage human coding skills in VSCode with Gemini Code Assist for generating diffs, refactors, and new code snippets. The focus is on iterative, testable changes to achieve a stable, self-improving CLI agent.

Estimated Effort: 4-6 hours of coding/refactoring, plus 1-2 hours testing.

#### **2. Problem Analysis**
Based on logs and architecture review:
- **Infinite Loops**: Replan -> Planner -> Tool Executor cycle exceeds recursion limit (25) due to repeated failures without termination.
- **Tool Brittleness**: `search_the_web` and `http_get` fail on rate limits/authentication; no retries or alternatives.
- **Planner Shortcomings**: LLM doesn't detect failure patterns, leading to repetitive plans; misses opportunities for dynamic tool creation.
- **State Management**: Growing `messages` list causes context overflow; no pruning or summarization.
- **Integration Gaps**: Memory Sub-Agent underutilized; no startup checks; dynamic tools not triggered aggressively.
- **Evaluation Deficit**: Lack of benchmarks allows regressions.

These stem from incomplete resilience in Phase 1 (correction loop) and Phase 4 (dynamic tools) of prior plans.

#### **3. Core Enhancements**
Adopt an "AI First" refactor: LLM orchestrates via improved prompts; Python enforces guards.

3.1. **Circuit Breaker Mechanism**
   - Add `replan_attempts: int` and `failed_actions: List[str]` to `state.py`.
   - In `replan` node (`main.py`): Increment counter; if >3, inject HumanMessage: "Escalating: Repeated failures detected. Call request_human_assistance with summary."
   - New Conditional Edge: After `replan`, if attempts >3, route to `handle_human_assistance`.
   - VSCode Action: Use Gemini to generate diff for `StateGraph` updates. Prompt: "Add circuit breaker to this LangGraph workflow with replan_attempts threshold."

3.2. **Advanced Error Handling**
   - Enhance `handle_error` (`main.py`): Parse tool output for HTTP codes/categories (e.g., auth vs. rate limit); append classified summary to `messages`.
   - Update `check_for_tool_error`: If error matches auth/rate patterns, set flag in state for forced replan strategy.
   - Prune `messages` in `replan`: Summarize history >10 messages into a single entry.
   - VSCode Action: Gemini refactor `handle_error`. Prompt: "Classify errors in this node and add message pruning logic."

3.3. **Planner and Replan Prompt Improvements**
   - Revise prompts to mandate failure analysis: "Analyze failed_actions list; if pattern (e.g., >2 API key fails), MUST plan tool creation: 1. search_the_web for code, 2. write_file to generated_tools/."
   - Force diversification: "Do not repeat failed tools; prioritize dynamic generation or assistance."
   - VSCode Action: Gemini generate new prompts. Prompt: "Enhance this planner prompt for failure pattern recognition and tool creation triggers."

3.4. **Tool Robustness and Dynamic Generation**
   - Update `tools.py`: Add retries (3x with delays) to `search_the_web` and `http_get`; rotate user-agents.
   - Ensure dynamic loading in `main.py` triggers on every cycle.
   - Auto-trigger creation for common failures (e.g., weather without key: generate scraper tool).
   - VSCode Action: Gemini update tools. Prompt: "Add retries and user-agent rotation to search_the_web in this file."

3.5. **Sub-Agent and Startup Checks**
   - In `main.py` CLI init: Ping MEMORY_SERVICE_URL; if fails, log warning and disable memory tools or fall back to local Chroma.
   - Integrate memory for plans: In `planner`, query memory for similar past failures.
   - VSCode Action: Gemini add checks. Prompt: "Insert sub-agent health check in this main.py init."

3.6. **Termination and Evaluation**
   - In `route_after_planner`: If no tools and content empty, end with failure report.
   - Implement Phase 5: Create `evaluate.py` and `evaluation/test_weather_failure.yaml` (success: "Report has weather data without loop").
   - Increase recursion_limit to 50 temporarily.
   - VSCode Action: Gemini generate evaluate.py. Prompt: "Build evaluate.py from v2.5 spec with weather YAML."

#### **4. Implementation Phases**
Phased rollout to minimize risk; test after each.

**Phase 1: State and Workflow Updates (1 hour)**
- Edit `state.py`: Add new fields.
- Refactor graph in `main.py`: Add circuit breaker edges/nodes.
- Test: Run weather query; ensure escalation after 3 fails.

**Phase 2: Prompt and Error Enhancements (1 hour)**
- Update prompts in `planner`/`replan`.
- Enhance `handle_error` with classification/pruning.
- Test: Simulate API fail; verify diversified plan.

**Phase 3: Tool Improvements (1 hour)**
- Refactor `tools.py` for robustness.
- Test: Manual tool calls; check retries.

**Phase 4: Integrations and Evaluation (1-2 hours)**
- Add startup checks.
- Build `evaluate.py` and YAML.
- Test: Full CLI run; eval pass/fail.

**Phase 5: Final Validation**
- Run indexer.py; verify DB.
- Monitor VRAM with nvidia-smi.
- Iterate on failures using Gemini assists.

#### **5. Risks and Mitigations**
- **Prompt Drift**: Review LLM outputs; tweak if repetitive.
- **VSCode Integration**: If Gemini assist fails, fallback to manual coding.
- **Backward Compatibility**: No breaking changes; test v2.5 features.
- **Scope Creep**: Stick to fixes; defer new features (e.g., voice mode).

This plan empowers human-AI collaboration: Use VSCode for code gen, review diffs, and implement. Post-completion, update to v3.0 in repo.