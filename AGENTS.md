# miniserve Agent Instructions
## Absolute rules
1. No performance claim without a committed CSV in bench/data/.
2. Modifying blocks.py or scheduler.py requires running its tests first.
3. Never add more than 200 lines of new code in one pass.
4. Never mark a task complete without running its verification command.
5. Correctness precedes performance; greedy parity vs HF is the gate.
## Verification commands
- Unit:        pytest tests/ -x -q
- Parity:      pytest tests/test_parity.py -x
- Bench smoke: python bench/run_suite.py --concurrency 1 --isl 128 --osl 32
