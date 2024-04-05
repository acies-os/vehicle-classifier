[private]
default:
    @just --list

# --connect unixsock-stream//tmp/acies-mic.sock \
# --connect unixsock-stream//tmp/acies-geo.sock \
vfm ns:
    LOGLEVEL=debug acies-vfm \
    --namespace {{ns}} \
    --proc_name vfm \
    --topic {{ns}}/geo \
    --topic {{ns}}/mic \
    --weight "models/Parkland_TransformerV4_vehicle_classification_1.0_finetune_ict15_latest.pt"
