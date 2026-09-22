#!/bin/zsh
# Re-create the embedding-viewer screenshots in this folder (macOS, Google Chrome installed).
# The viewer itself is unmodified: a throwaway copy gets a small driver appended that loads a
# run's checkpoint.json through the viewer's own "Open your checkpoint" file input, then picks
# the Before / After / Show-movement view. Run from the repository root:
#   zsh experiments/embedding_viewer/capture_viewer.sh
set -e
OUT=experiments/embedding_viewer TMP=$(mktemp -d)
CH="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
cp embedding-viewer.html viewer_capture.tmp.html
cat >> viewer_capture.tmp.html <<'EOF'
<script>
(async()=>{
  const url=new URLSearchParams(location.search).get('ckpt');
  const dt=new DataTransfer();
  dt.items.add(new File([await (await fetch(url)).blob()],'checkpoint.json',{type:'application/json'}));
  const input=document.getElementById('checkpoint'); input.files=dt.files; input.dispatchEvent(new Event('change'));
  await new Promise(r=>setTimeout(r,800));
  if(location.hash==='#before') document.getElementById('before').click();
  if(location.hash==='#trails'){const t=document.getElementById('trails'); t.checked=true; t.dispatchEvent(new Event('change'));}
})();
</script>
EOF
.venv/bin/python -m http.server 8765 --bind 127.0.0.1 >/dev/null 2>&1 & SERVER=$!
trap 'kill $SERVER 2>/dev/null; rm -f viewer_capture.tmp.html; rm -rf $TMP' EXIT
sleep 1.5
for run in starter expanded; do for view in before after trails; do
  "$CH" --headless=new --disable-gpu --hide-scrollbars --no-first-run --window-size=1440,1000 \
    --force-device-scale-factor=2 --virtual-time-budget=6000 --user-data-dir=$TMP/$run$view \
    --screenshot=$TMP/${run}_${view}.png "http://127.0.0.1:8765/viewer_capture.tmp.html?ckpt=experiments/$run/run/checkpoint.json#$view" \
    >/dev/null 2>&1 & CHROME=$!
  for i in {1..40}; do [[ -s $TMP/${run}_${view}.png ]] && sleep 1 && break; sleep 1; done
  kill $CHROME 2>/dev/null || true
  .venv/bin/python -c "from PIL import Image; Image.open('$TMP/${run}_${view}.png').convert('RGB').crop((40,330,2840,1690)).resize((1400,680),Image.LANCZOS).save('$OUT/${run}_${view}.png',optimize=True)"
done; done
echo "Saved screenshots to $OUT/"
