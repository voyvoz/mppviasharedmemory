# Build

```
cargo build --release
```

# After Building

Requires

```
setcap cap_net_admin+ep <target_bin>
```

# Options

```
<target_bin> -h
```

to see available options.

# Running

Use the `script/setup.sh` to add an IP range, disable IPv6, set the correct MTU and set the link online.

# Note

When stopping the program you need to run the setup script again after starting it.
