if [ ! $# -eq 2 ]; then
    echo "$0 <car-name> <car-ip>"
    exit 1
fi

CAR_IP=$2
CAR_HOSTNAME=$1

echo "$CAR_IP  $CAR_HOSTNAME" >> /etc/hosts

cat <<EOL >> ~/.ssh/config

Host $CAR_HOSTNAME
        User nvidia
EOL
