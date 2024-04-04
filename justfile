[private]
default:
    @just --list

vfm:
    LOGLEVEL=debug acies-vfm \
    --namespace "rs1" \
    --proc_name "vfm" \
    --weight "models/Parkland_TransformerV4_vehicle_classification_1.0_finetune_ict15_latest.pt"
