# Connecting the PC and the Jetson

The scripts in `host/` reach the Jetson over SSH through one setting,
`JETSON_HOST` in `host/jetson.env` (copy `host/jetson.env.example`). It can be
an `~/.ssh/config` alias (recommended), `user@ip`, or `user@hostname.local`.
The Jetson must be reachable without a password prompt (SSH key) for the
scripts to run unattended.

## Option A: the Jetson on your normal network

Plug the Jetson into the same router as your PC. JetPack runs Avahi, so
`<jetson-hostname>.local` usually resolves without any setup:

```bash
ssh-copy-id <user>@<jetson-hostname>.local   # once, asks for the password
echo 'JETSON_HOST=<user>@<jetson-hostname>.local' > host/jetson.env
```

## Option B: a direct cable between the PC and the Jetson (what this project uses)

Useful when the Jetson has no other network, and it gives the Jetson internet
through the PC. On the PC (NetworkManager), create a "shared" connection on
the Ethernet port the cable is plugged into:

```bash
nmcli connection add type ethernet ifname <eth-port> con-name jetson-share \
    ipv4.method shared            # the PC becomes 10.42.0.1 and runs DHCP + NAT
```

**The trap: boot order.** If the Jetson boots before the PC has brought this
connection up, its DHCP request times out and NetworkManager on the Jetson may
fall back to an old lease from another network (seen here: a stale
`192.168.137.x` lease from a Windows "Internet Connection Sharing" setup). The
Jetson is then on the cable but unreachable. The robust fix is a static
address on the Jetson's Ethernet, so boot order does not matter (run on the
Jetson):

```bash
sudo nmcli connection modify "Wired connection 1" ipv4.method manual \
    ipv4.addresses 10.42.0.24/24 ipv4.gateway 10.42.0.1 ipv4.dns 10.42.0.1
sudo nmcli connection up "Wired connection 1"
```

(If you also plug that port into a normal router at times, add a second
connection profile with `ipv4.method auto` instead of changing this one.)

To find a Jetson that "disappeared" on the cable, look at what the PC sees on
that port: `ip neigh show dev <eth-port>` (NVIDIA MAC addresses start with
`00:04:4b`), or `cat /var/lib/NetworkManager/dnsmasq-<eth-port>.leases`.

## The SSH alias

`~/.ssh/config` on the PC:

```
Host jetson
    HostName 10.42.0.24          # or <jetson-hostname>.local
    User <user>
    IdentityFile ~/.ssh/<your-jetson-key>
    IdentitiesOnly yes
    ServerAliveInterval 30
```

Then `ssh jetson`, `host/sync_to_nano.sh` and the other host scripts work with
the default `JETSON_HOST=jetson`.

## Ports

- `22`: SSH.
- `8090`: the MJPEG stream and the dashboard (`nano/apps/*`). The dashboard has
  no authentication: keep it on a trusted network, or start it with
  `--bind 127.0.0.1` and tunnel: `ssh -L 8090:localhost:8090 jetson`, then
  open `http://localhost:8090/` on the PC.
