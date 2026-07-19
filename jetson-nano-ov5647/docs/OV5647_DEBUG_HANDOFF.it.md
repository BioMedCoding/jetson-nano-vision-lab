# OV5647 su Jetson Nano — Handoff debug camera (stato al 2026-07-18)

> **RISOLTO (2026-07-18, sessione successiva)** — la camera cattura frame e l'intera
> pipeline CV funziona (trt_detect a 30 FPS live). Vedi la sezione "RISOLUZIONE" in fondo
> per le cause e i fix. Il resto del documento è mantenuto come storico del debug.

Questo file è pensato per essere l'UNICA fonte necessaria per riprendere il lavoro da zero,
senza dover rileggere la conversazione originale. Contiene: setup, obiettivo, tutto ciò che
è stato fatto (funzionante e non), le fonti trovate online, lo stato attuale esatto del
sistema, e le idee per il prossimo passo.

## 1. Setup — come riconnettersi e dove sono le cose

**Jetson Nano** (target):
```
ssh -i ~/.ssh/jetson_session/jetson_key matteo@10.42.0.24
```
Per comandi che richiedono `sudo` (serve un TTY interattivo — NON eseguibile da tool non
interattivi, va lanciato nel terminale reale dell'utente):
```
ssh -t -i ~/.ssh/jetson_session/jetson_key matteo@10.42.0.24
```
Hostname target: `matteo-desktop`. Kernel attuale in esecuzione: 4.9.253, build `#15` (numero
incrementa a ogni reboot con nuova Image — controllare `uname -a` per sapere quale build gira).

**Sorgenti kernel** (host, dove si compila):
- Root L4T: `/home/matteo/jetson_work/Linux_for_Tegra/source/public/`
- Kernel "vanilla" (contiene `drivers/media/i2c/ov5647.c`, il driver del sensore):
  `.../kernel/kernel-4.9/`
- Driver Tegra VI/CSI/camera_common (NVIDIA proprietario ma sorgente disponibile):
  `.../kernel/nvidia/drivers/media/platform/tegra/camera/`
  - `vi/channel.c` — logica canale VI (format negotiation, stream on/off, mode setup)
  - `vi/vi2_fops.c` — VI2 fops (T210/Nano usa VI2+CSI2, NON VI4/VI5/nvcsi che sono per
    Xavier/Orin — attenzione a non confondersi leggendo csi4/csi5/vi4/vi5 fops, non sono
    questa piattaforma)
  - `csi/csi.c` — logica CSI comune, inclusa `tegra_csi_mipi_calibrate()`
  - `csi/csi2_fops.c` — fops CSI2 specifiche T210, incluso `csi2_mipi_cal()` e
    `csi2_start_streaming()`
  - `camera_common.c` — framework camera_common (`camera_common_initialize()` ecc.)
  - `.../include/media/camera_common.h` — contiene `to_camera_common_data()` (vedi fix #1)
- Device tree del sensore:
  `.../hardware/nvidia/platform/t210/porg/kernel-dts/porg-platforms/tegra210-camera-ov5647.dtsi`
- Sensore ufficiale funzionante di riferimento (per confronto): `drivers/media/i2c/imx219.c`
  (usa tegracam completo, non usato da noi — vedi sezione 4)

**Build** (sempre da `.../kernel/kernel-4.9/`):
```bash
export CROSS_COMPILE=aarch64-linux-gnu- ARCH=arm64
make -j$(nproc) Image                          # kernel Image (~10s, incrementale)
make -j$(nproc) M=drivers/media/i2c modules    # solo ov5647.ko (~1-2s)
make -j$(nproc) dtbs                           # tutti i DTB inclusi il nostro
```
`.config` esiste già nell'albero, non serve rigenerarlo.

**Deploy sul target** (via `scp` + comandi sudo interattivi dell'utente):
- Image → `/boot/Image-ov5647` (referenziata dalla entry di default in
  `/boot/extlinux/extlinux.conf`, LABEL già esistente, non serve toccare extlinux.conf)
- Modulo → `/lib/modules/4.9.253/kernel/drivers/media/i2c/ov5647.ko` + poi `sudo depmod -a`
- DTB → `/boot/dtb/tegra210-p3448-0000-p3449-0000-b00-ov5647.dtb` (path confermato leggendo
  `FDT` in extlinux.conf)
- Serve **sempre un reboot** dopo aver toccato Image o DTB (non solo il modulo). Il modulo da
  solo NON si può ricaricare a caldo: `rmmod ov5647` fallisce con "Module is in use" — il VI
  driver (built-in) tiene un riferimento persistente al subdev anche a fd chiuso, non c'è un
  meccanismo di unbind pulito lato VI in questo driver. Reboot completo è l'unica via
  affidabile trovata.

**Pattern di deploy usato per tutta la sessione** (funziona, ripetibile):
1. Build in locale.
2. `scp` verso `~/qualcosa.{ko,dtb}` o `~/Image-xxx` sul target (mai sovrascrivere
   direttamente `/boot/...` via scp, permessi negati — copiare prima in home).
3. Dare all'utente 2-4 comandi da eseguire **nel suo terminale reale** (non tool):
   `ssh -t ...` poi `sudo cp ...` poi (se serve) `sudo depmod -a` poi `sudo reboot`.
   **Importante**: NON concatenare i comandi con `&&` in una riga sola passata a `ssh -t
   "..."` — il terminale dell'utente a volte va a capo su comandi lunghi e spezza la riga,
   facendo eseguire il path di destinazione di `cp` come comando a sé (errore visto più
   volte: "cp: missing destination file operand" seguito da "bash: line 1:
   /percorso/del/file: Permission denied"). Dare comandi brevi, uno per riga.
4. Attendere il reboot pollando (l'host è già raggiungibile ~5s dopo il reboot):
   ```bash
   SSH="ssh -i ~/.ssh/jetson_session/jetson_key -o BatchMode=yes -o ConnectTimeout=5 -o StrictHostKeyChecking=accept-new matteo@10.42.0.24"
   for i in $(seq 1 30); do $SSH "echo up" 2>/dev/null | grep -q up && break; sleep 5; done
   ```

**Script di test cattura** (sul target, persistente): `~/v4l2_raw_test3.py`. Fa ioctl V4L2 raw
via `fcntl.ioctl` (no OpenCV/GStreamer, bypassa Argus completamente): apre `/dev/video0`,
imposta `VIDIOC_S_CTRL` per forzare `TEGRA_CAMERA_CID_VI_BYPASS_MODE=1`, poi
`S_FMT(640x480, BG10)` → `REQBUFS` → `QUERYBUF`+`mmap`+`QBUF` ×4 → `STREAMON` → tenta 5
`DQBUF` con timeout 2s ciascuno → `STREAMOFF`. Se cattura frame reali, salva 4 varianti di
demosaic Bayer (`bggr2bgr`, `rggb2bgr`, `gbrg2bgr`, `grbg2bgr`) in `/tmp/ov5647_*.jpg` sul
target, da scaricare e ispezionare visivamente per capire l'ordine Bayer corretto (questo
passo non è mai stato raggiunto — non abbiamo ancora catturato un solo frame).
Per rieseguirlo: `source ~/cv_projects/venv/bin/activate && python3 ~/v4l2_raw_test3.py`.

**Ambiente Python sul target**: venv a `~/cv_projects/venv` (Python 3.6 --system-site-packages,
richiesto per compatibilità con le wheel torch/torchvision cp36 usate altrove nel progetto).

## 2. Obiettivo originale

Il progetto più ampio (`cv_projects/` sulla SD, vedi memoria `jetson_cv_projects_setup`) è
avere un rilevatore YOLOv5 funzionante su Jetson Nano con una camera OV5647 (Raspberry Pi
Camera V1, clone). Il collo di bottiglia attuale è: **la camera non cattura nemmeno un
frame**, indipendentemente dal software applicativo (Argus/`nvarguscamerasrc`, GStreamer,
OpenCV, o raw V4L2 diretto — tutti falliscono allo stesso modo a livello driver).

Nota: Argus/`nvarguscamerasrc` è stato escluso definitivamente come strada percorribile — è
uno stack closed-source con whitelist di sensori fissa, non riconoscerà mai un sensore non
"tegracam" indipendentemente da qualunque fix nel driver V4L2. Tutto il lavoro successivo è
su **V4L2 raw**, bypassando Argus.

## 3. Le hardware/config di riferimento

- Jetson Nano devkit originale (non 2GB), L4T R32.7.1, kernel 4.9.253, JetPack 4.6.1.
- Camera: OV5647 (Raspberry Pi Camera V1 clone), connettore a 15 pin, **collegato
  DIRETTAMENTE** al connettore CAM0 della Jetson Nano, **senza adattatori**. L'utente ha
  confermato esplicitamente questo — NON è un problema di adattatore/cablaggio pin-mismatch
  (avevo sollevato questa ipotesi confondendo la Jetson Nano con la Jetson TX1, che ha un
  connettore CSI diverso e richiede davvero un adattatore 15↔22 pin secondo la doc RidgeRun
  per TX1 — quell'informazione NON si applica alla Nano, che accetta il cavo 15 pin
  direttamente nel suo slot 22 pin allineato correttamente, configurazione standard e ben
  documentata per questo esatto abbinamento hardware).
- Driver di partenza: `drivers/media/i2c/ov5647.c` — il driver mainline Linux "grezzo"
  (Ramiro Oliveira/Synopsys), NON integrato con `camera_common`/`tegracam` di NVIDIA. Questo
  è il cuore del problema: gli unici sensori con cui NVIDIA testa la propria stack VI/CSI
  sono quelli integrati con `camera_common`, e questo driver non lo era affatto all'inizio
  della sessione.

## 4. Cosa è stato risolto (confermato, stabile)

### 4.1 Crash del kernel al probe (RISOLTO)
**Sintomo**: OOPS ("Internal error: Accessing user space memory outside uaccess.h routines")
in `tegra_channel_populate_dev_info` al probe del sensore.

**Causa**: `to_camera_common_data()` in `camera_common.h` faceva `container_of()` sul
`drvdata` di QUALSIASI device il cui nodo DT avesse figli `modeN`, assumendo ciecamente che
fosse un vero `camera_common_data`. Il nostro `ov5647.c` (prima del fix del §4.3) lasciava
nel drvdata il proprio `v4l2_subdev` grezzo, non un `camera_common_data` — la
reinterpretazione di quella memoria come struttura diversa causava dereference di puntatori
spazzatura.

**Fix** (file: `kernel/nvidia/include/media/camera_common.h`, funzione
`to_camera_common_data()`): aggiunta verifica di auto-riferimento — dopo il
`container_of()`, controlla che `s_data->dev == dev`; se non corrisponde, ritorna `NULL`
invece di fidarsi ciecamente. `camera_common_initialize()` imposta sempre `->dev` al device
del sensore stesso, quindi questo self-check distingue un vero consumer camera_common da uno
che non lo è.

```c
static inline struct camera_common_data *to_camera_common_data(
	const struct device *dev)
{
	struct camera_common_data *s_data;

	if (!sensor_common_parse_num_modes(dev))
		return NULL;

	s_data = container_of(dev_get_drvdata(dev),
			    struct camera_common_data, subdev);
	if (s_data->dev != dev)
		return NULL;

	return s_data;
}
```
**Stato**: confermato stabile per l'intera sessione (decine di reboot, mai più ricomparso).

### 4.2 Negoziazione formato V4L2 rotta (RISOLTO)
**Sintomo**: `VIDIOC_S_FMT` sembrava avere successo ma non cambiava nulla; dopo molto
debug si è scoperto che in realtà falliva silenziosamente con `-ENOTTY`, mascherato per
ore da un bug nostro (vedi §5.1).

**Causa reale**: `ov5647.c` non implementava `.pad->get_fmt`/`.pad->set_fmt` nel
`v4l2_subdev_pad_ops` — solo `.enum_mbus_code`. Senza questi, `tegra_channel_try_format`/
`set_format` in `channel.c` (che chiamano `v4l2_subdev_call(sd, pad, set_fmt, ...)`) fallivano
con `-ENOIOCTLCMD`→`-ENOTTY`, e la ioctl veniva instradata al default handler
`tegra_channel_default_ioctl` che no-oppa silenziosamente.

**Fix**: aggiunte `ov5647_get_fmt`/`ov5647_set_fmt` che riportano sempre la risoluzione
fissa realmente trasmessa (inizialmente 640×480 SBGGR8, poi migrato a SBGGR10 — vedi §4.4),
agganciate a `.pad->get_fmt`/`.pad->set_fmt`.

**Stato**: confermato funzionante — `REQBUFS`/`QUERYBUF` allocano correttamente buffer da
307200 byte (640×480×1, versione 8-bit) o 614400 byte (640×480×2, versione 10-bit attuale).

### 4.3 Calibrazione PHY MIPI mai eseguita (TROVATO, integrato — non ancora risolutivo)
**Sintomo**: dopo aver risolto §4.1 e §4.2, `STREAMON` ha successo ma `DQBUF` va sempre in
timeout; dmesg mostra ripetutamente `video4linux video0: frame start syncpt timeout!0`,
senza nessun altro errore (nessun errore I2C, nessun errore CSI PHY esplicito).

**Causa trovata** (file `kernel/nvidia/drivers/media/platform/tegra/camera/csi/csi.c`,
funzione `tegra_csi_mipi_calibrate()`):
```c
list_for_each_entry(chan, &csi->csi_chans, list) {
    if (chan->pg_mode) continue;
    if (chan->s_data == NULL) continue;   // <-- QUESTO
    ret = csi->fops->mipical(chan);
    ...
}
```
`chan->s_data` è popolato da `to_camera_common_data(chan->subdev_on_csi->dev)` — che per il
nostro sensore (mai stato un vero consumer camera_common) tornava sempre `NULL` (correttamente,
per via del fix del §4.1). Risultato: **la calibrazione del PHY MIPI veniva saltata
silenziosamente** per il nostro canale, per l'intera sessione fino a questo fix. Un ricevitore
mai calibrato può restare armato ma incapace di agganciare qualunque transizione HS valida —
compatibile al 100% con "zero frame, zero errori" osservato.

Confermato con un `pr_info` diagnostico temporaneo (poi rimosso da `csi2_fops.c`, ma un
secondo più permanente è rimasto in `csi.c`, vedi §6): prima del fix, `s_data=NULL`,
`sensor_sd=NULL`, la funzione di calibrazione non veniva mai chiamata. Dopo il fix,
`s_data`/`sensor_sd` sono validi (indirizzi non-NULL) e `fops->mipical()` ritorna
esplicitamente `0` (successo, verificato anche leggendo l'implementazione reale in
`mipical/mipi_cal.c` — non è uno stub, esegue davvero `select_lanes()` +
`tegra_mipi_wait()` con controllo errore genuino).

**Fix implementato** (file: `kernel/kernel-4.9/drivers/media/i2c/ov5647.c`): integrazione
MINIMA con `camera_common` — non il framework `tegracam` completo (che richiederebbe
conversione a `regmap`, `ctrl_ops` per esposizione/gain, `sensor_ops` — troppo lavoro per
quello che serve), ma uso diretto delle API di basso livello:
- `struct ov5647` non contiene più il proprio `struct v4l2_subdev sd` incorporato; contiene
  invece `struct camera_common_data *s_data`.
- `to_state()` ora recupera la struct driver via
  `container_of(sd, struct camera_common_data, subdev)->priv`.
- In `ov5647_probe()`: si alloca `s_data`, si popolano i campi minimi (`dev`, `priv`,
  `frmfmt`/`numfmts` con una entry 640×480, `def_mode=0`, `def_width=640`, `def_height=480`,
  `def_clk_freq=24000000`), si chiama `camera_common_initialize(s_data, "ov5647")` (che fa
  `camera_common_parse_ports()` + `camera_common_parse_general_properties()` +
  `sensor_common_init_sensor_properties()` leggendo i nodi DT `modeN` esistenti), poi si
  imposta `s_data->mode_prop_idx = 0`, e si registra `&s_data->subdev` (non più un subdev
  a parte) via `v4l2_i2c_subdev_init()`.
- `ov5647_remove()` aggiornato di conseguenza.

**Stato**: l'integrazione compila pulita, non causa crash, e la calibrazione MIPI ora
**viene davvero eseguita e ritorna successo** (verificato via debug print, vedi sopra).
Nonostante questo, **il frame start continua a non arrivare** — vedi §5 per cosa è stato
escluso dopo.

### 4.4 Register table del sensore migrata a 10-bit (fatto, non risolutivo da solo)
Vedi §5.3 per il ragionamento — comunque tenuto perché è la versione più corretta/verificata
disponibile (porting fedele dal driver mainline Linux attualmente mantenuto, testato da
terze parti su hardware reale). Non riportato indietro a 8-bit.

### 4.5 Dati del modo0 nel device tree corretti (fatto, non risolutivo da solo)
File: `hardware/nvidia/platform/t210/porg/kernel-dts/porg-platforms/tegra210-camera-ov5647.dtsi`,
nodo `mode0`. Originariamente descriveva una modalità 3264×2464@21fps/182.4MHz completamente
fittizia (mai realmente trasmessa da questo driver, che implementa UNA SOLA modalità fissa
640×480 via la sua register table). Corretto con i valori reali del nostro stream (presi dal
driver upstream verificato): `active_w=640`, `active_h=480`, `pixel_t=bayer_bggr`,
`line_length=1852` (= HTS), `pix_clk_hz=58333000`, `embedded_metadata_height=0`. Gli altri 5
nodi `modeN` (mode1-mode5) sono rimasti invariati e ancora fittizi/inutilizzati — non
influenzano nulla perché `mode_prop_idx` punta sempre a 0.

## 5. Cosa è stato provato e NON ha funzionato (con perché sembrava promettente)

Tutti questi test sono stati eseguiti CON il fix §4.1+4.2 già applicato e stabile, e (dal
punto in cui indicato) con anche il fix §4.3 (camera_common) applicato. Ogni singolo
tentativo qui sotto ha prodotto l'IDENTICO sintomo: `STREAMON` ok, poi
`frame start syncpt timeout!0` ripetuto, zero errori aggiuntivi nei log, zero frame mai
ricevuti.

### 5.1 (Storico, già risolto ma menzionato per contesto) Bug nostro nel test Python
Per molte ore il vero bug (§4.2, ENOTTY mascherato) è stato nascosto da un errore banale
nostro: calcolo sbagliato della dimensione di `struct v4l2_format` nello script di test
Python (204 byte invece di 208 — la union interna del formato è arrotondata a 8 byte per
allineamento struct). Con l'ioctl code sbagliato, `__video_do_ioctl` non riconosceva
`VIDIOC_S_FMT` e lo instradava al default handler, che no-oppava silenziosamente — dando
l'illusione che "S_FMT abbia successo ma non faccia nulla". Trovato con un `pr_info`
incondizionato in cima a `video_usercopy()` (kernel-4.9/drivers/media/v4l2-core/v4l2-ioctl.c)
che stampava fianco a fianco il `cmd` ricevuto e la costante `VIDIOC_S_FMT` calcolata dal
kernel stesso — la differenza (`0xc0cc5605` vs `0xc0d05605`) ha rivelato l'errore di
dimensione. **Lezione**: quando si fanno ioctl raw a mano in Python, ricalcolare sempre con
precisione `sizeof()` delle struct C reali, non fidarsi di conteggi manuali dei campi.

### 5.2 Compensazione PLL per clock 24MHz invece di 25MHz (NON HA RISOLTO)
**Ipotesi**: il driver controlla esplicitamente `if (xclk_freq != 25000000)
dev_warn("unusual xclk frequency")` — e stampa questo warning ad ogni boot, perché il clock
MCLK fisico della Jetson (fisso a livello di board, condiviso da tutti i sensori camera, vedi
`assigned-clock-rates` per `TEGRA210_CLK_EXTERN3` nel DT di board-level, NON nel dtsi del
sensore) è 24MHz, non i 25MHz per cui l'intera register table del sensore (inclusi
moltiplicatore/divisori PLL) era stata originariamente calibrata. Il registro `0x4802`
(controllo timing D-PHY) ha default `0x00` = calcolo automatico di TUTTI i parametri di
timing HS/CLK basato sul clock PLL reale prodotto — quindi un PLL sbagliato del 4%
propagherebbe l'errore a ogni singolo parametro di timing MIPI.

**Fix tentato**: compensazione proporzionale del moltiplicatore PLL (registro `0x3036`):
`nuovo = round(vecchio × 25/24)`. Con `vecchio=0x46=70` → `nuovo=73=0x49`.

**Risultato**: nessun cambiamento, stesso identico fallimento.

**Perché non era la causa (probabile)**: si è poi scoperto (vedendo la register table del
driver mainline attualmente mantenuto, vedi §5.3) che il valore di `0x3036=0x46` è IDENTICO
tra la loro versione 10-bit e la nostra versione 8-bit originale — cioè il moltiplicatore PLL
non cambia tra le due modalità nel driver reale e funzionante, suggerendo che il PLL di per sé
non sia il problema (o quantomeno che la mia compensazione proporzionale, seppur
matematicamente ragionevole, non colga il vero meccanismo, dato che il driver upstream non fa
nulla di simile e presumibilmente funziona su altre board con XVCLK diverso da 25MHz senza
questo tipo di correzione).

### 5.3 Porting completo a 10-bit dal driver mainline attivamente mantenuto (NON HA RISOLTO da solo — ma tenuto)
**Ipotesi**: ricerca online (vedi §7) ha mostrato che TUTTE le implementazioni
OV5647-su-Jetson funzionanti trovate (RidgeRun commerciale per Nano E per TX1, un progetto
community "giraffai" scritto da zero con integrazione Tegracam) usano RAW10 (10-bit), MAI
RAW8. Il nostro driver di partenza (`drivers/media/i2c/ov5647.c`, versione storica nel
repo Linux mainline, autore Ramiro Oliveira/Synopsys) usa invece `MEDIA_BUS_FMT_SBGGR8_1X8`
(8-bit) con registro `0x3034=0x08` — un valore che, controllando il datasheet ufficiale
OV5647 (sezione 6.1, tabella system control registers), **non corrisponde nemmeno a un
valore documentato** per il campo `mipi_bit_mode` (bit[3:0] di `0x3034`): solo `0000`=8bit e
`0001`=10bit sono definiti, "altri: riservati per uso futuro". `0x08` in binario ha
bit[3:0]=`1000`, che non è né l'uno né l'altro.

**Fix tentato**: portato fedelmente da
`https://raw.githubusercontent.com/torvalds/linux/master/drivers/media/i2c/ov5647.c`
(versione attuale mainline, autore aggiornato, usa `struct reg_sequence` + `regmap` — noi
abbiamo tradotto i valori registro nella nostra struttura `regval_list` più semplice,
mantenendo la logica di scrittura raw i2c esistente) la loro modalità
"`ov5647_640x480_10bpp`" (commento originale: "10-bit VGA full FOV 60fps. 2x2 binned and
subsampled down to VGA", `.pixel_rate=58333000`, `.hts=1852`, `.vts=0x1f8`=504). Cambiato
`MEDIA_BUS_FMT_SBGGR8_1X8`→`MEDIA_BUS_FMT_SBGGR10_1X10` in `get_fmt`/`set_fmt`/
`enum_mbus_code`/`ov5647_open`. Verificato che la tabella formati Tegra
(`vi/vi2_formats.h`) ha già un entry nativo per `SBGGR10_1X10`→`RAW10` CSI datatype, quindi
nessuna modifica lato Tegra necessaria per questo. Script di test aggiornato per
`fourcc('BG10')` e buffer a 2 byte/pixel.

**Risultato**: `REQBUFS`/`QUERYBUF` ora allocano correttamente 614400 byte (640×480×2,
conferma che la negoziazione formato regge anche col nuovo formato) — ma stesso identico
`frame start syncpt timeout`. Nessun miglioramento.

**Stato**: il codice è stato TENUTO (non riportato a 8-bit) perché è comunque la versione
più corretta/verificata disponibile, indipendentemente dal fatto che non abbia risolto da
sola il problema principale.

### 5.4 Deskew delle lane CSI (VERIFICATO non rilevante)
`tegra_channel_set_stream()` in `channel.c` ha un secondo meccanismo di calibrazione,
`nvcsi_deskew_apply_check()`, attivato se `!chan->bypass && !chan->pg_mode &&
chan->deskew_ctx->deskew_lanes`. Verificato via debug print: `deskew_ctx` non è NULL ma
`deskew_lanes=0` — quindi il branch non si attiva mai. Inoltre `nvcsi_deskew_apply_check` è
comunque uno stub che ritorna sempre 0 su T210 se `CONFIG_TEGRA_GRHOST_NVCSI` non è abilitato
(deskew è una feature nvcsi, tipicamente T194+/Xavier, non T210/Nano). **Escluso come causa.**

### 5.5 Modalità bypass VI forzata esplicitamente (NON HA RISOLTO)
**Ipotesi**: `chan->bypass` (controllo V4L2 `TEGRA_CAMERA_CID_VI_BYPASS_MODE`, default
`.def=0`=OFF) determina se il canale VI scrive i dati RAW direttamente in memoria
(bypass=1, quello che vogliamo per cattura raw) o se passa per un percorso più
ISP-consapevole (bypass=0). Verificato via debug print che durante tutti i test precedenti
`bypass=0` (mai esplicitamente impostato dal nostro script di test finale — un tentativo
precedente, poi abbandonato, provava a impostarlo ma non era mai stato re-integrato nello
script definitivo). Ipotesi: forse il canale con bypass=0 si aspetta segnali/config lato ISP
che il nostro sensore raw non fornisce mai, bloccando il frame-start.

**Fix tentato**: aggiunta chiamata `VIDIOC_S_CTRL` esplicita nello script di test
(`~/v4l2_raw_test3.py`) per impostare `TEGRA_CAMERA_CID_VI_BYPASS_MODE=1` (indice 1 =
`SWITCH_ON` nel `switch_ctrl_qmenu`) PRIMA di `S_FMT`/`STREAMON`.

**Risultato**: `S_CTRL` ha successo (`set VI_BYPASS_MODE=1 (SWITCH_ON) ok`), confermato via
debug print che `bypass=1` durante il successivo `set_stream(on=1)` — ma stesso identico
`frame start syncpt timeout`. Nessun cambiamento.

## 6. Stato ESATTO del sistema in questo momento

**Sul target** (dopo l'ultimo reboot, build kernel `#15`):
- `/boot/Image-ov5647` = build con: fix §4.1 (camera_common.h) + fix §4.2 (get/set_fmt) +
  fix §4.3 (integrazione camera_common in ov5647.c, compilata nell'Image perché
  `camera_common.c`/`channel.c`/`csi.c`/`csi2_fops.c` sono `obj-y`, built-in) + **due
  `pr_info` diagnostici ancora presenti**:
  - `kernel/nvidia/drivers/media/platform/tegra/camera/csi/csi.c`, dentro
    `tegra_csi_mipi_calibrate()`, stampa `CLAUDE_DBG mipi_calibrate: chan=... pg_mode=...
    s_data=... sensor_sd=... numlanes=...` e poi `CLAUDE_DBG mipi_calibrate: fops->mipical
    returned %d`.
  - `kernel/nvidia/drivers/media/platform/tegra/camera/vi/channel.c`, dentro
    `tegra_channel_set_stream()`, stampa `CLAUDE_DBG set_stream: on=... bypass=...
    pg_mode=... deskew_ctx=...` e (se non NULL) `CLAUDE_DBG set_stream: deskew_lanes=...`.
  
  Questi print sono **volutamente lasciati** perché utili per continuare il debug — NON sono
  stati rimossi. Se il problema viene risolto, vanno tolti prima di considerare il lavoro
  concluso (cercare `CLAUDE_DBG` in entrambi i file).
- `/lib/modules/4.9.253/kernel/drivers/media/i2c/ov5647.ko` = versione con register table
  10-bit (§5.3) + integrazione camera_common (§4.3). Formato dichiarato: SBGGR10_1X10,
  640×480 fisso.
- `/boot/dtb/tegra210-p3448-0000-p3449-0000-b00-ov5647.dtb` = con `mode0` corretto (§4.5).
- `~/v4l2_raw_test3.py` = versione con forzatura bypass=1 (§5.5) + fourcc BG10 + FMT_SIZE=208
  corretto. Questo è lo script di test più aggiornato.

**In locale** (host, sorgenti — riflettono esattamente quanto sopra):
- `kernel/kernel-4.9/drivers/media/i2c/ov5647.c` — driver completo come descritto in §4.3+§4.4.
- `kernel/nvidia/include/media/camera_common.h` — fix §4.1 presente.
- `kernel/nvidia/drivers/media/platform/tegra/camera/csi/csi.c` — debug print presente (§6).
- `kernel/nvidia/drivers/media/platform/tegra/camera/vi/channel.c` — debug print presente (§6).
- `hardware/.../tegra210-camera-ov5647.dtsi` — mode0 corretto (§4.5).

**Ultimo risultato di test** (identico da diversi tentativi):
```
set VI_BYPASS_MODE=1 (SWITCH_ON) ok
S_FMT -> 640 640 ...  (nota: il parsing del nostro script Python di questo specifico print
                        ha probabilmente un bug di allineamento struct nel debug output —
                        REQBUFS/QUERYBUF sono corretti e affidabili, quindi la negoziazione
                        formato REALE è verificata giusta anche se questa riga sembra strana)
G_FMT (chan->format) -> 0 640 ...
REQBUFS -> count= 4
QUERYBUF 0..3 -> length= 614400   (= 640*480*2, corretto per 10-bit)
STREAMON ok
timeout waiting for frame 0
```
dmesg mostra: calibrazione MIPI eseguita con successo (`fops->mipical returned 0`),
`bypass=1` confermato, poi ripetuti `frame start syncpt timeout!0`, nessun altro errore.

## 7. Ricerca online — cosa è stato trovato

- **Forum NVIDIA developer** (thread "Mipi-driver : frame start syncpt timeout!0",
  3 pagine): stesso identico sintomo per un sensore custom su Jetson. **Mai risolto in modo
  confermato** nel thread — ha divagato su verifiche hardware (collegamento pin, clock) senza
  conclusione. Altri thread NVIDIA sullo stesso errore ("Got frame start syncpt timeout!0",
  "Nano camera capture error", ecc.) — pattern ricorrente, nessuna soluzione definitiva
  pubblica trovata per un caso equivalente al nostro.
- **`Moore123/ov5647_jetson`** (GitHub): tentativo di driver OV5647 per Jetson con
  integrazione tegracam (`tegracam_core.h`), esplicitamente marcato **"Not Working Yet"**
  nel README.
- **RidgeRun** (`developer.ridgerun.com`, azienda di consulenza embedded seria):
  vende driver OV5647 commerciali per Jetson Nano E per Jetson TX1, **entrambi
  esplicitamente RAW10/Bayer BGGR 10-bit**, mai RAW8. Nessun prezzo pubblico, tarball delle
  patch fornita solo dietro contatto commerciale diretto (non liberamente scaricabile).
  Nessun dettaglio tecnico pubblico su come risolvono il timing CSI/PLL.
- **Blog "giraffai.com"**, serie in 5 parti "Developing OV5647 camera sensor Linux device
  driver for Jetson Nano": l'unico resoconto pubblico di qualcuno che ha scritto un driver
  OV5647 DA ZERO con vera integrazione Tegracam e **ottenuto frame reali** funzionanti
  (comando citato: `v4l2-ctl --set-fmt-video=... pixel=RG10 ... --set-ctrl bypass_mode=0`
  — nota: loro usano `bypass_mode=0`, NON 1 come la nostra ipotesi §5.5; inconsistenza mai
  chiarita). Non documenta i dettagli tecnici precisi di come hanno risolto il timing
  CSI/frame-sync — l'autore rimanda esplicitamente a "un video dimostrativo" invece di
  spiegarlo per iscritto.
- **`torvalds/linux` mainline `drivers/media/i2c/ov5647.c`** (versione attuale, molto più
  recente della nostra): usata come fonte per il porting §5.3. Conferma `xclk` deve essere
  esattamente 25MHz (ora è un **errore fatale**, non solo un warning, nelle versioni più
  recenti). Usa `regmap`/`cci_*` invece di i2c raw. Nessuna logica di calcolo PLL dinamico
  per XVCLK diversi da 25MHz — il problema "adattare per un XVCLK diverso" non risulta
  risolto nemmeno upstream.
- **Datasheet ufficiale OV5647** (OmniVision, "PRELIMINARY SPECIFICATION", versione 1.0,
  novembre 2009 — scaricato e letto pagina per pagina): confermato "input clock frequency:
  6~27 MHz" (quindi 24MHz è ampiamente nel range supportato dal sensore in teoria), "PLL...
  genera un clock interno di default a 96 MHz da un input 6-27MHz" (implica che il PLL
  dovrebbe normalizzare l'output indipendentemente dall'input, se configurato correttamente
  per quello specifico input). Tabella 6-1 (system control registers, pag. 6-3/6-4) ha dato i
  bit-field esatti di `0x3034`-`0x3037` (PLL charge pump, mipi_bit_mode, system_clk_div,
  scale_divider_mipi, PLL multiplier 4-252, pll_root_div, pll_prediv) usati per derivare (poi
  rivelatosi insufficiente da solo) il fix §5.2.

## 8. Idee non ancora testate per il prossimo tentativo

In ordine di costo/informazione stimati (dal più economico/informativo al più costoso):

1. **Leggere i registri di stato CSI reali al momento del timeout** (mai fatto). File
   `csi/csi2_fops.c` ha già gli helper `csi_read()`/`cil_read()` usati ovunque per scrivere
   `TEGRA_CSI_PIXEL_PARSER_STATUS`, `TEGRA_CSI_CIL_STATUS`, `TEGRA_CSI_CILX_STATUS` (viste
   scritte con `0xFFFFFFFF` per "pulire" lo stato in `csi2_start_streaming()` — quindi sono
   leggibili e significative). Aggiungere un `pr_info` che LEGGE questi registri subito dopo
   il fallimento di `nvhost_syncpt_wait_timeout_ext()` in `vi2_fops.c` (riga ~443-454, dove
   già stampa "frame start syncpt timeout") darebbe il vero stato hardware del ricevitore
   D-PHY: se ha visto ANCHE solo una transizione LP→HS (probabile problema di framing/CRC,
   più vicino a un bug software/timing fine) o se non ha visto assolutamente NULLA (molto più
   indicativo di un problema elettrico/fisico o di configurazione lane/porta CSI errata).
   Questo è probabilmente il singolo prossimo passo con miglior rapporto
   informazione/rischio.

2. **Provare la Test Pattern Generator (TPG) integrata di Tegra** (mai fatto). Il driver VI2
   supporta `chan->pg_mode` — un percorso che genera dati sintetici internamente al SoC,
   completamente indipendente dal sensore fisico e dal cavo. Se il TPG produce frame validi
   attraverso lo stesso canale VI/CSI, isolerebbe definitivamente il problema al sensore o
   al cavo (non a Tegra); se anche il TPG fallisce con lo stesso timeout, indicherebbe un
   problema di configurazione VI/CSI indipendente dal sensore (cosa mai considerata finora).
   Vedere `graph.c`/`channel.c` per come si attiva pg_mode (probabilmente serve una entry DT
   `tpg` dedicata, o via debugfs).

3. **Verificare l'assegnazione fisica porta/lane CSI** (mai verificato a fondo). Jetson Nano
   ha porte CSI multiple (CAM0/CAM1) mappate a NVCSI port A/B con routing lane specifico.
   Se il DT del sensore dichiara la porta CSI sbagliata rispetto al connettore fisico
   realmente usato, la calibrazione e la ricezione avverrebbero sulle lane fisiche
   sbagliate — compatibile con "tutto configurato correttamente ma zero segnale mai
   ricevuto". Controllare `csi_port`/`reg` nei nodi DT (`vi_port0`, `csi_chan0`, ecc. in
   `tegra210-camera-ov5647.dtsi`) contro il connettore fisico effettivamente usato
   dall'utente (CAM0 vs CAM1, se la board ne ha più di uno).

4. **Confrontare riga per riga con `digitallyamar/ov5647`** (GitHub, trovato ma mai
   esaminato in dettaglio — `https://github.com/digitallyamar/ov5647`). Repository più
   recente della ricerca iniziale, non ancora controllato per contenuto tecnico/stato.

5. **Contattare RidgeRun per un preventivo** (opzione commerciale, non tecnica) — hanno
   dichiarato un driver funzionante per Jetson Nano specificamente. Se il budget lo permette
   in futuro, potrebbe risolvere direttamente bypassando l'intera indagine.

6. **Riconsiderare hardware**: un IMX219 (Raspberry Pi Camera V2, supportato nativamente da
   Jetson Nano, zero lavoro driver) resta l'opzione più pragmatica se l'obiettivo primario è
   "avere una camera funzionante per YOLOv5" piuttosto che "far funzionare questo specifico
   sensore". L'utente ha detto di non poter acquistare ora — da riconsiderare se il punto 1-4
   non sblocca nulla.

## RISOLUZIONE (2026-07-18, sessione successiva)

Il passo §8.1 (lettura registri di stato CSI al timeout) è stato eseguito SENZA rebuild:
il percorso di recovery del timeout chiama già `tegra_channel_capture_error()` →
`tegra_csi_status()` che stampa esattamente i registri voluti, solo in `dev_dbg`; con
`CONFIG_DYNAMIC_DEBUG=y` (già attivo) basta:
`echo 'file csi2_fops.c +p' | sudo tee /sys/kernel/debug/dynamic_debug/control` (idem
`vi2_fops.c`), rilanciare il test e leggere dmesg.

**Diagnosi**: `PIXEL_PARSER_STATUS=0x0` sempre (mai decodificato un pacchetto), ma
`CIL_STATUS=0x10` (CILA_CTRL_ERR: sequenze LP abortite in LP11) e
`CILX_STATUS=0x00040041` (CTRL_ERR su clock lane, bit0 + data lane 0, bit6 + data lane 1,
bit18). Quindi: il sensore trasmetteva, ma il ricevitore non agganciava mai LP→HS.

**Causa 1 — clock MIPI**: il DT dichiara `discontinuous_clk="yes"` ma `__sensor_init()`
scriveva `0x4800=0x04` (clock lane gate OFF → clock CONTINUO), sovrascrivendo lo 0x34
della register table; inoltre il sensore usciva dal power-on già in streaming (mai una
transizione LP11→HS osservabile). Fix in `ov5647.c` (come mainline `clock-noncontinuous`):
stream_on → `0x4800=0x34` (gate+line sync+bus idle); stream_off → `0x4800=0x25` (lane in
LP-11); `__sensor_init()` termina con `ov5647_stream_off()`.

**Causa 2 — bypass invertito (correzione di §5.5)**: `TEGRA_CAMERA_CID_VI_BYPASS_MODE`
deve essere **0**: con bypass=1 `vi2_channel_start_streaming()` ritorna subito dopo il
set_stream del sensore — niente kthread di cattura, niente syncpt armato (per questo in
quelle run non c'era nemmeno il messaggio di timeout). giraffai aveva ragione.

**Causa 3 — frame neri**: la table 10-bit mainline usa `0x3503=0x03` (AEC/AGC manuali,
pilotati da controlli V4L2 che questo driver non ha) → esposizione/gain ~0. Fix:
`0x3503=0x00` (auto). L'AEC converge in ~2-3 s di streaming continuo.

**Ordine Bayer reale: GRBG** (verificato con oggetto rosso: corretto con
`cv2.COLOR_BayerGR2BGR`; driver/DT dichiarano BGGR, etichetta innocua per la pipeline raw).

**Pipeline CV (tutti gli script testati funzionanti)**:
- `common/camera.py` riscritto su V4L2 raw ioctl (backup: `camera.py.argus_backup`);
  interfaccia identica (read/isOpened/release), demosaic GRBG, ~62.5 fps.
- 01 camera_test: 62.5 fps; 02 live_detect (torch FP16 yolov5n@416): 17.8 fps;
  03 trt_detect (engine FP16 416, buildato): **30.6 fps live**; 04 benchmark: TRT 23.6ms
  vs PyTorch FP16 43.9ms @416; 05 capture_images ok (usare --width 640 --height 480,
  altre risoluzioni = resize software).
- Fix ambiente fatti sul target: seaborn import opzionale in `~/yolov5/utils/plots.py`
  (poi yolov5 ha auto-installato seaborn 0.11.2 nel venv); `~/.config/Ultralytics/Arial.ttf`
  (copia di DejaVuSans, l'URL di download è morto); pycuda 2021.1 installato con
  `--no-build-isolation` (la build isolation compilava numpy 1.12 → errore xlocale.h);
  `LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1` per cv2-dopo-torch (aggiunto in
  build_engine.sh; serve anche per lanciare trt_detect/benchmark).

**Cleanup residuo (non bloccante)**: rimuovere i due `pr_info` CLAUDE_DBG (csi.c,
channel.c) dall'Image; eventualmente allineare l'etichetta formato (SGRBG10 nel driver +
`pixel_t=bayer_grbg` nel DT). `~/v4l2_raw_test3.py` ora cattura 300 frame (bypass=0).

## 9. Nota su affidabilità/precisione di questo documento

Tutti i fatti su codice, path, valori di registro, e risultati di test in questo documento
sono stati verificati direttamente (lettura sorgenti, esecuzione su hardware reale, lettura
dmesg) durante la sessione — non sono ipotesi non testate, salvo dove esplicitamente
indicato come "ipotesi"/"non ancora testato". I risultati di ricerca web (§7) sono riassunti
da fonti esterne e vanno ri-verificati se usati come base per nuove decisioni (i link erano
validi al momento della ricerca, 2026-07-18).
