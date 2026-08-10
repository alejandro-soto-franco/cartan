# cartan-homog-valid

Validation harness for `cartan-homog`. ECHOES-generated NPZ fixtures + Rust tests.

## Fixture layout

- `fixtures/basic/v1/` — committed basic set (< 100 KB total). Always runs.
- `$CARTAN_HOMOG_FIXTURES_DIR/v1/` — extended set, held outside the tree. Skipped silently if unavailable.

## Regenerating fixtures

```bash
cd cartan-homog-valid/python       # a python 3.12 env with the echoes wheel
export CARTAN_HOMOG_FIXTURES_DIR=/path/to/homog-fixtures
python generate_fixtures.py --config configs/v1_test_matrix.yaml \
    --out "$CARTAN_HOMOG_FIXTURES_DIR/v1" \
    --mirror-basic
```
