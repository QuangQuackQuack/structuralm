#### TRAIN ####
python launch.py --config configs/dreamfusion-if.yaml --train --gpu 0 system.prompt_processor.prompt="a large bed" system.background.random_aug=true
##### INFERENCE ####
python launch.py --config outputs/dreamfusion-if/a_large_bed@20251122-225933/configs/parsed.yaml --export --gpu 0 resume=outputs/dreamfusion-if/a_large_bed@20251122-225933/ckpts/last.ckpt system.exporter_type=mesh-exporter system.exporter.context_type=cuda system.exporter.fmt=obj-mtl

### NOTE: Nhớ thay tên folder vào scripts inference (ví dụ như dưới)
a_large_bed@20251122-225933



