# On your laptop – package everything once
DOTS=~/Library/Application\ Support/Cursor/User   # macOS, adjust for Linux: ~/.config/Cursor/User
tar czf cursor_dotfiles.tgz -C "$DOTS" settings.json keybindings.json snippets

# Copy to RunPod
# scp cursor_dotfiles.tgz mu8l026yugnukv-6441117a@ssh.runpod.io:/tmp

# On RunPod – unpack where VS Code server looks
ssh [mu8l026yugnukv-6441117a@ssh.runpod.io] '
  mkdir -p ~/.vscode-server/data/User
  tar xzf /tmp/cursor_dotfiles.tgz -C ~/.vscode-server/data/User
'