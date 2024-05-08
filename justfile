[private]
default:
    @just --list

clean:
    rm -f *.log

ns := `hostname -s`
router_ip := "10.8.0.3"

vfm-weight-2 := "models/demo2024_Parkland_TransformerV4_vehicle_classification_1.0_finetune_yizhuoict15_best.pt"
vfm-weight-geo := "models/demo2024_Parkland_TransformerV4_vehicle_classification_1.0_finetune_yizhuoict15_seismic_best.pt"
vfm-weight-mic := "models/demo2024_Parkland_TransformerV4_vehicle_classification_1.0_finetune_yizhuoict15_audio_best.pt"

mae-weight-2 := "models/demo2024_freqmae_Parkland_TransformerV4_vehicle_classification_1.0_finetune_best.pt"

vfm:
    LOGLEVEL=debug rye run acies-vfm \
    --connect unixsock-stream//tmp/{{ns}}_acies-mic.sock \
    --connect unixsock-stream//tmp/{{ns}}_acies-geo.sock \
    --connect tcp/{{router_ip}}:7447 \
    --namespace {{ns}} \
    --proc_name vfm \
    --topic {{ns}}/geo \
    --topic {{ns}}/mic \
    --weight {{vfm-weight-2}}

deactivated-vfm:
    LOGLEVEL=debug rye run acies-vfm \
    --connect unixsock-stream//tmp/{{ns}}_acies-mic.sock \
    --connect unixsock-stream//tmp/{{ns}}_acies-geo.sock \
    --connect tcp/{{router_ip}}:7447 \
    --namespace {{ns}} \
    --proc_name vfm \
    --topic {{ns}}/geo \
    --topic {{ns}}/mic \
    --deactivated \
    --weight {{vfm-weight-2}}

backup-vfm ns_primary:
    LOGLEVEL=debug rye run acies-vfm \
    --connect unixsock-stream//tmp/{{ns_primary}}_acies-mic.sock \
    --connect unixsock-stream//tmp/{{ns_primary}}_acies-geo.sock \
    --connect tcp/{{router_ip}}:7447 \
    --namespace {{ns}} \
    --proc_name "backup/{{ns_primary}}/vfm" \
    --topic {{ns_primary}}/geo \
    --topic {{ns_primary}}/mic \
    --deactivated \
    --weight {{vfm-weight-2}}

backup-vfm-geo ns_primary:
    LOGLEVEL=debug rye run acies-vfm \
    --connect unixsock-stream//tmp/{{ns_primary}}_acies-mic.sock \
    --connect unixsock-stream//tmp/{{ns_primary}}_acies-geo.sock \
    --connect tcp/{{router_ip}}:7447 \
    --namespace {{ns}} \
    --proc_name "backup/{{ns_primary}}/vfm_geo" \
    --modality 'seismic' \
    --topic {{ns_primary}}/geo \
    --topic {{ns_primary}}/mic \
    --deactivated \
    --weight {{vfm-weight-geo}}

backup-vfm-mic ns_primary:
    LOGLEVEL=debug rye run acies-vfm \
    --connect unixsock-stream//tmp/{{ns_primary}}_acies-mic.sock \
    --connect unixsock-stream//tmp/{{ns_primary}}_acies-geo.sock \
    --connect tcp/{{router_ip}}:7447 \
    --namespace {{ns}} \
    --proc_name "backup/{{ns_primary}}/vfm_mic" \
    --modality 'audio' \
    --topic {{ns_primary}}/geo \
    --topic {{ns_primary}}/mic \
    --deactivated \
    --weight {{vfm-weight-mic}}

mae:
    LOGLEVEL=debug rye run acies-vfm \
    --connect unixsock-stream//tmp/{{ns}}_acies-mic.sock \
    --connect unixsock-stream//tmp/{{ns}}_acies-geo.sock \
    --connect tcp/{{router_ip}}:7447 \
    --namespace {{ns}} \
    --proc_name mae \
    --topic {{ns}}/geo \
    --topic {{ns}}/mic \
    --weight {{mae-weight-2}}

ds:
    LOGLEVEL=debug rye run acies-ds \
    --connect unixsock-stream//tmp/{{ns}}_acies-mic.sock \
    --connect unixsock-stream//tmp/{{ns}}_acies-geo.sock \
    --connect tcp/{{router_ip}}:7447 \
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

start:
    just {{ns}}

end:
    pkill -SIGINT -f 'acies-vfm'

rs1:
    nohup just vfm > nohup_vfm.log 2>&1 &
    nohup just backup-vfm rs2 > nohup_vfm.log 2>&1 &
    nohup just backup-vfm-geo rs1 > nohup_vfm_geo.log 2>&1 &
    nohup just backup-vfm-mic rs1 > nohup_vfm_mic.log 2>&1 &

rs2:
    nohup just vfm > nohup_vfm.log 2>&1 &
    nohup just backup-vfm rs3 > nohup_vfm.log 2>&1 &
    nohup just backup-vfm-geo rs2 > nohup_vfm_geo.log 2>&1 &
    nohup just backup-vfm-mic rs2 > nohup_vfm_mic.log 2>&1 &

rs3:
    nohup just vfm > nohup_vfm.log 2>&1 &
    nohup just backup-vfm rs5 > nohup_vfm.log 2>&1 &
    nohup just backup-vfm-geo rs3 > nohup_vfm_geo.log 2>&1 &
    nohup just backup-vfm-mic rs3 > nohup_vfm_mic.log 2>&1 &

rs5:
    nohup just vfm > nohup_vfm.log 2>&1 &
    nohup just backup-vfm rs6 > nohup_vfm.log 2>&1 &
    nohup just backup-vfm-geo rs5 > nohup_vfm_geo.log 2>&1 &
    nohup just backup-vfm-mic rs5 > nohup_vfm_mic.log 2>&1 &

rs6:
    nohup just vfm > nohup_vfm.log 2>&1 &
    nohup just backup-vfm rs7 > nohup_vfm.log 2>&1 &
    nohup just backup-vfm-geo rs6 > nohup_vfm_geo.log 2>&1 &
    nohup just backup-vfm-mic rs6 > nohup_vfm_mic.log 2>&1 &

rs7:
    nohup just vfm > nohup_vfm.log 2>&1 &
    nohup just backup-vfm rs8 > nohup_vfm.log 2>&1 &
    nohup just backup-vfm-geo rs7 > nohup_vfm_geo.log 2>&1 &
    nohup just backup-vfm-mic rs7 > nohup_vfm_mic.log 2>&1 &

rs8:
    nohup just vfm > nohup_vfm.log 2>&1 &
    nohup just backup-vfm rs10 > nohup_vfm.log 2>&1 &
    nohup just backup-vfm-geo rs8 > nohup_vfm_geo.log 2>&1 &
    nohup just backup-vfm-mic rs8 > nohup_vfm_mic.log 2>&1 &

rs10:
    nohup just vfm > nohup_vfm.log 2>&1 &
    nohup just backup-vfm rs1 > nohup_vfm.log 2>&1 &
    nohup just backup-vfm-geo rs10 > nohup_vfm_geo.log 2>&1 &
    nohup just backup-vfm-mic rs10 > nohup_vfm_mic.log 2>&1 &
