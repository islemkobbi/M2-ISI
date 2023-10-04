#!/bin/bash


python -m src.main --train --evaluate --batch_size 8 --lr 1.0e-4 --steps 20000 --q
python -m src.main --train --evaluate --batch_size 8 --lr 1.0e-3 --steps 20000 --q
python -m src.main --train --evaluate --batch_size 8 --lr 5.0e-3 --steps 20000 --q
python -m src.main --train --evaluate --batch_size 16 --lr 1.0e-4 --steps 20000 --q
python -m src.main --train --evaluate --batch_size 16 --lr 1.0e-3 --steps 20000 --q
python -m src.main --train --evaluate --batch_size 16 --lr 5.0e-3 --steps 20000 --q
python -m src.main --train --evaluate --batch_size 32 --lr 1.0e-4 --steps 20000 --q
python -m src.main --train --evaluate --batch_size 32 --lr 1.0e-3 --steps 20000 --q
python -m src.main --train --evaluate --batch_size 32 --lr 5.0e-3 --steps 20000 --q
python -m src.main --train --evaluate --batch_size 64 --lr 1.0e-4 --steps 20000 --q
python -m src.main --train --evaluate --batch_size 64 --lr 1.0e-3 --steps 20000 --q
python -m src.main --train --evaluate --batch_size 64 --lr 5.0e-3 --steps 20000 --q