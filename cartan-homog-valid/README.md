# cartan-homog-valid

Validation harness for `cartan-homog`. ECHOES-generated NPZ fixtures + Rust tests.

## Fixture layout

- `fixtures/basic/v1/` — committed basic set (< 100 KB total). Always runs.
- `$CARTAN_HOMOG_FIXTURES_DIR/v1/` — external extended set (ASF-EX2 default). Skipped silently if unavailable.

## Regenerating fixtures

```bash
conda activate echoes-homog        # python 3.12 env with echoes wheel installed
cd cartan-homog-valid/python
export CARTAN_HOMOG_FIXTURES_DIR=/run/media/alejandrosotofranco/ASF-EX2/cartan/homog-fixtures
python generate_fixtures.py --config configs/v1_test_matrix.yaml \
    --out "$CARTAN_HOMOG_FIXTURES_DIR/v1" \
    --mirror-basic
```
