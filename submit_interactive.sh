
runai submit heichole-cropping \
  -i aicregistry:5000/mhuber:heichole_cropping \
  -g 1 \
  --interactive \
  -v /nfs:/nfs \
  --large-shm \
  -- sleep infinity
