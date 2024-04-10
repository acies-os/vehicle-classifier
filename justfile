[private]
default:
    @just --list

clean:
    rm -f *.log

vfm ns:
    LOGLEVEL=debug acies-vfm \
    --connect unixsock-stream//tmp/{{ns}}_acies-mic.sock \
    --connect unixsock-stream//tmp/{{ns}}_acies-geo.sock \
    --connect tcp/10.8.0.3:7447 \
    --namespace {{ns}} \
    --proc_name vfm \
    --topic {{ns}}/geo \
    --topic {{ns}}/mic \
    --weight "models/demo2024_Parkland_TransformerV4_vehicle_classification_1.0_finetune_yizhuoict15_best.pt"

deactivated-vfm ns:
    LOGLEVEL=debug acies-vfm \
    --connect unixsock-stream//tmp/{{ns}}_acies-mic.sock \
    --connect unixsock-stream//tmp/{{ns}}_acies-geo.sock \
    --connect tcp/10.8.0.3:7447 \
    --namespace {{ns}} \
    --proc_name vfm \
    --topic {{ns}}/geo \
    --topic {{ns}}/mic \
    --deactivated \
    --weight "models/demo2024_Parkland_TransformerV4_vehicle_classification_1.0_finetune_yizhuoict15_best.pt"

backup-vfm ns_carrier ns_primary:
    LOGLEVEL=debug acies-vfm \
    --connect unixsock-stream//tmp/{{ns_primary}}_acies-mic.sock \
    --connect unixsock-stream//tmp/{{ns_primary}}_acies-geo.sock \
    --connect tcp/10.8.0.3:7447 \
    --namespace {{ns_carrier}} \
    --proc_name "backup/{{ns_primary}}/vfm" \
    --topic {{ns_primary}}/geo \
    --topic {{ns_primary}}/mic \
    --deactivated \
    --weight "models/demo2024_Parkland_TransformerV4_vehicle_classification_1.0_finetune_yizhuoict15_best.pt"

backup-vfm-geo ns_carrier ns_primary:
    LOGLEVEL=debug acies-vfm \
    --connect unixsock-stream//tmp/{{ns_primary}}_acies-mic.sock \
    --connect unixsock-stream//tmp/{{ns_primary}}_acies-geo.sock \
    --connect tcp/10.8.0.3:7447 \
    --namespace {{ns_carrier}} \
    --proc_name "backup/{{ns_primary}}/vfm_geo" \
    --modality 'seismic' \
    --topic {{ns_primary}}/geo \
    --topic {{ns_primary}}/mic \
    --deactivated \
    --weight "models/demo2024_Parkland_TransformerV4_vehicle_classification_1.0_finetune_yizhuoict15_seismic_best.pt"

backup-vfm-mic ns_carrier ns_primary:
    LOGLEVEL=debug acies-vfm \
    --connect unixsock-stream//tmp/{{ns_primary}}_acies-mic.sock \
    --connect unixsock-stream//tmp/{{ns_primary}}_acies-geo.sock \
    --connect tcp/10.8.0.3:7447 \
    --namespace {{ns_carrier}} \
    --proc_name "backup/{{ns_primary}}/vfm_mic" \
    --modality 'audio' \
    --topic {{ns_primary}}/geo \
    --topic {{ns_primary}}/mic \
    --deactivated \
    --weight "models/demo2024_Parkland_TransformerV4_vehicle_classification_1.0_finetune_yizhuoict15_audio_best.pt"

mae ns:
    LOGLEVEL=debug acies-vfm \
    --connect unixsock-stream//tmp/{{ns}}_acies-mic.sock \
    --connect unixsock-stream//tmp/{{ns}}_acies-geo.sock \
    --connect tcp/10.8.0.3:7447 \
    --namespace {{ns}} \
    --proc_name mae \
    --topic {{ns}}/geo \
    --topic {{ns}}/mic \
    --weight "models/demo2024_freqmae_Parkland_TransformerV4_vehicle_classification_1.0_finetune_best.pt"

ds ns:
    LOGLEVEL=debug acies-ds \
    --connect unixsock-stream//tmp/{{ns}}_acies-mic.sock \
    --connect unixsock-stream//tmp/{{ns}}_acies-geo.sock \
    --connect tcp/10.8.0.3:7447 \
    --namespace {{ns}} \
    --proc_name ds \
    --topic {{ns}}/geo \
    --topic {{ns}}/mic \
    --model-config "models/demo2024_deepsense_multi.yaml" \
    --weight "models/demo2024_deepsense_train_best.pt"

push-weight:
    #!/usr/bin/env fish
    for i in rs1 rs2 rs3 rs5 rs6 rs6 rs7 rs8 rs10
        rsync -azvhP models/demo2024* $i.dev:/ws/acies/vehicle-classifier/models
    end
