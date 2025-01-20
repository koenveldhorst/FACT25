# Usage: bash ssh_connect.sh ssh_key_name
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/$1
ssh -T git@github.com