
runai submit heichole-cropping \
  -i aicregistry:5000/mhuber:heichole_cropping \
  -g 0 \
  --interactive \
  -v /nfs:/nfs \
  --large-shm \
  --run-as-user \
  -- sleep infinity
