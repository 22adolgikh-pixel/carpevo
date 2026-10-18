Код из MobileSAM (https://github.com/ChaoningZhang/MobileSAM) и Segment Anything (https://github.com/facebookresearch/segment-anything),
обе — Apache License 2.0. Изменения для CPS: убраны зависимости timm/torchvision (to_2tuple, trunc_normal_, DropPath — свои),
из sam.py убран ImageEncoderViT (используется только TinyViT). Веса mobile_sam.pt скачиваются при первом запуске (sam_seg.py).
