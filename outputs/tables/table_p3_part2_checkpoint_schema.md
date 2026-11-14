| Saved Component        | Data Structure               | Scope                             | Verification                              |
|:-----------------------|:-----------------------------|:----------------------------------|:------------------------------------------|
| Model State Dict       | OrderedDict[Tensor]          | All trainable & buffer parameters | Bitwise match upon reload                 |
| Optimizer State Dict   | Dict[ParamGroup, State]      | Momentum & variance buffers       | Bitwise state preservation                |
| Scheduler State Dict   | Dict[step, base_lrs]         | Epoch count and step counters     | Seamless trajectory resumption            |
| GradScaler State Dict  | Dict[scale, growth_tracker]  | Dynamic loss scale value          | Scale continuation without re-calibration |
| RNG Seed States        | Torch, Numpy & Python states | Pseudo-random generator streams   | Exact deterministic replay                |
| Cryptographic Checksum | SHA-256 Digest String        | Entire binary .pt file content    | Zero bit-rot, tamper detection            |