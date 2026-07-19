/*
 * A V4L2 driver for OmniVision OV5647 cameras.
 *
 * Based on Samsung S5K6AAFX SXGA 1/6" 1.3M CMOS Image Sensor driver
 * Copyright (C) 2011 Sylwester Nawrocki <s.nawrocki@samsung.com>
 *
 * Based on Omnivision OV7670 Camera Driver
 * Copyright (C) 2006-7 Jonathan Corbet <corbet@lwn.net>
 *
 * Copyright (C) 2016, Synopsys, Inc.
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License as
 * published by the Free Software Foundation version 2.
 *
 * This program is distributed .as is. WITHOUT ANY WARRANTY of any
 * kind, whether express or implied; without even the implied warranty
 * of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 */

#include <linux/clk.h>
#include <linux/delay.h>
#include <linux/i2c.h>
#include <linux/init.h>
#include <linux/io.h>
#include <linux/module.h>
#include <linux/of_graph.h>
#include <linux/slab.h>
#include <linux/videodev2.h>
#include <media/v4l2-device.h>
#include <media/v4l2-of.h>
#include <media/v4l2-image-sizes.h>
#include <media/v4l2-mediabus.h>
#include <media/camera_common.h>

#define SENSOR_NAME "ov5647"

#define OV5647_SW_RESET		0x0103
#define OV5647_REG_CHIPID_H	0x300A
#define OV5647_REG_CHIPID_L	0x300B

#define REG_TERM 0xfffe
#define VAL_TERM 0xfe
#define REG_DLY  0xffff

#define OV5647_ROW_START		0x01
#define OV5647_ROW_START_MIN		0
#define OV5647_ROW_START_MAX		2004
#define OV5647_ROW_START_DEF		54

#define OV5647_COLUMN_START		0x02
#define OV5647_COLUMN_START_MIN		0
#define OV5647_COLUMN_START_MAX		2750
#define OV5647_COLUMN_START_DEF		16

#define OV5647_WINDOW_HEIGHT		0x03
#define OV5647_WINDOW_HEIGHT_MIN	2
#define OV5647_WINDOW_HEIGHT_MAX	2006
#define OV5647_WINDOW_HEIGHT_DEF	1944

#define OV5647_WINDOW_WIDTH		0x04
#define OV5647_WINDOW_WIDTH_MIN		2
#define OV5647_WINDOW_WIDTH_MAX		2752
#define OV5647_WINDOW_WIDTH_DEF		2592

struct regval_list {
	u16 addr;
	u8 data;
};

struct ov5647 {
	/*
	 * The registered v4l2_subdev is s_data->subdev, not an
	 * ov5647-owned field: this is required so that
	 * to_camera_common_data() (container_of on ->subdev) resolves
	 * to a real, valid camera_common_data for this device. Without
	 * it, chan->s_data stays NULL for this sensor's CSI channel and
	 * tegra_csi_mipi_calibrate() silently skips MIPI PHY calibration
	 * for it -- the receiver is then never calibrated and no frame
	 * start is ever detected, regardless of what the sensor sends.
	 */
	struct camera_common_data	*s_data;
	struct media_pad		pad;
	struct mutex			lock;
	struct v4l2_mbus_framefmt	format;
	unsigned int			width;
	unsigned int			height;
	int				power_count;
	struct clk			*xclk;
};

static inline struct ov5647 *to_state(struct v4l2_subdev *sd)
{
	struct camera_common_data *s_data =
		container_of(sd, struct camera_common_data, subdev);
	return (struct ov5647 *)s_data->priv;
}

static struct regval_list sensor_oe_disable_regs[] = {
	{0x3000, 0x00},
	{0x3001, 0x00},
	{0x3002, 0x00},
};

static struct regval_list sensor_oe_enable_regs[] = {
	{0x3000, 0x0f},
	{0x3001, 0xff},
	{0x3002, 0xe4},
};

/*
 * This 640x480 mode is ported from the actively-maintained upstream
 * Linux driver (drivers/media/i2c/ov5647.c, "2x2 binned and
 * subsampled VGA, 10-bit"), replacing the 8-bit table this file
 * originally shipped with. Three independent working OV5647-on-Jetson
 * ports (RidgeRun's Nano/TX1 drivers, and a from-scratch community
 * driver) all stream 10-bit RAW10, never 8-bit -- and 0x3034=0x08
 * used by the old 8-bit table here is not even a documented
 * mipi_bit_mode value in the OV5647 datasheet's own register table
 * (only 0x0 "8 bit" and 0x1 "10 bit" are defined in bits[3:0]).
 * Critically, upstream leaves MIPI global timing register 0x4837
 * untouched (auto-calculated from register 0x4802, whose reset
 * default already selects full auto-calculate for every D-PHY HS/CLK
 * timing parameter) instead of forcing an explicit value the way the
 * old 8-bit table here did -- a manually forced 0x4837 tuned for a
 * different assumed pixel clock is a plausible reason the old table
 * never produced a CSI frame-start on this platform's 24MHz sensor
 * clock (see ov5647_sensor_power()'s "unusual xclk frequency"
 * warning: this sensor is normally fed 25MHz, but Jetson's camera
 * MCLK is fixed board-wide at 24MHz).
 */
static struct regval_list ov5647_640x480[] = {
	{0x0100, 0x00},
	{0x0103, 0x01},
	{0x3034, 0x1a},
	{0x3035, 0x21},
	{0x303c, 0x11},
	{0x3106, 0xf5},
	{0x3827, 0xec},
	{0x370c, 0x03},
	{0x5000, 0x06},
	{0x5003, 0x08},
	{0x5a00, 0x08},
	{0x3000, 0x00},
	{0x3001, 0x00},
	{0x3002, 0x00},
	{0x3016, 0x08},
	{0x3017, 0xe0},
	{0x3018, 0x44},
	{0x301c, 0xf8},
	{0x301d, 0xf0},
	{0x3a18, 0x00},
	{0x3a19, 0xf8},
	{0x3c01, 0x80},
	{0x3b07, 0x0c},
	{0x3630, 0x2e},
	{0x3632, 0xe2},
	{0x3633, 0x23},
	{0x3634, 0x44},
	{0x3636, 0x06},
	{0x3620, 0x64},
	{0x3621, 0xe0},
	{0x3600, 0x37},
	{0x3704, 0xa0},
	{0x3703, 0x5a},
	{0x3715, 0x78},
	{0x3717, 0x01},
	{0x3731, 0x02},
	{0x370b, 0x60},
	{0x3705, 0x1a},
	{0x3f05, 0x02},
	{0x3f06, 0x10},
	{0x3f01, 0x0a},
	{0x3a08, 0x01},
	{0x3a0f, 0x58},
	{0x3a10, 0x50},
	{0x3a1b, 0x58},
	{0x3a1e, 0x50},
	{0x3a11, 0x60},
	{0x3a1f, 0x28},
	/*
	 * AGC gain ceiling 15.5x (0x0f8/16): without it the sensor default
	 * caps auto-gain so low that AEC plateaus far below its own target
	 * (0x3a0f/0x3a10) in anything but bright light.
	 */
	{0x3a18, 0x00},
	{0x3a19, 0xf8},
	{0x4001, 0x02},
	{0x4000, 0x09},
	/*
	 * AEC/AGC auto (mainline uses manual, 0x03, but drives exposure and
	 * gain via V4L2 controls this driver does not implement — manual
	 * with the power-on defaults produces black frames).
	 */
	{0x3503, 0x00},
	/* mode-specific: 640x480, 2x2 binned + subsampled, 10-bit */
	{0x3036, 0x46},
	{0x3821, 0x03},
	{0x3820, 0x41},
	{0x3612, 0x59},
	{0x3618, 0x00},
	{0x3814, 0x35},
	{0x3815, 0x35},
	{0x3708, 0x64},
	{0x3709, 0x52},
	{0x3800, 0x00},
	{0x3801, 0x10},
	{0x3802, 0x00},
	{0x3803, 0x00},
	{0x3804, 0x0a},
	{0x3805, 0x2f},
	{0x3806, 0x07},
	{0x3807, 0x9f},
	{0x3808, 0x02},
	{0x3809, 0x80},
	{0x380a, 0x01},
	{0x380b, 0xe0},
	/* HTS=1852 (0x073c), VTS=504 (0x01f8) */
	{0x380c, 0x07},
	{0x380d, 0x3c},
	{0x380e, 0x01},
	{0x380f, 0xf8},
	{0x3a09, 0x2e},
	{0x3a0a, 0x00},
	{0x3a0b, 0xfb},
	{0x3a0d, 0x02},
	{0x3a0e, 0x01},
	{0x4004, 0x02},
	{0x4800, 0x34},
	{0x0100, 0x01},
};

static int ov5647_write(struct v4l2_subdev *sd, u16 reg, u8 val)
{
	int ret;
	unsigned char data[3] = { reg >> 8, reg & 0xff, val};
	struct i2c_client *client = v4l2_get_subdevdata(sd);

	ret = i2c_master_send(client, data, 3);
	if (ret < 0)
		dev_dbg(&client->dev, "%s: i2c write error, reg: %x\n",
				__func__, reg);

	return ret;
}

static int ov5647_read(struct v4l2_subdev *sd, u16 reg, u8 *val)
{
	int ret;
	unsigned char data_w[2] = { reg >> 8, reg & 0xff };
	struct i2c_client *client = v4l2_get_subdevdata(sd);

	ret = i2c_master_send(client, data_w, 2);
	if (ret < 0) {
		dev_dbg(&client->dev, "%s: i2c write error, reg: %x\n",
			__func__, reg);
		return ret;
	}

	ret = i2c_master_recv(client, val, 1);
	if (ret < 0)
		dev_dbg(&client->dev, "%s: i2c read error, reg: %x\n",
				__func__, reg);

	return ret;
}

static int ov5647_write_array(struct v4l2_subdev *sd,
				struct regval_list *regs, int array_size)
{
	int i, ret;

	for (i = 0; i < array_size; i++) {
		ret = ov5647_write(sd, regs[i].addr, regs[i].data);
		if (ret < 0)
			return ret;
	}

	return 0;
}

static int ov5647_set_virtual_channel(struct v4l2_subdev *sd, int channel)
{
	u8 channel_id;
	int ret;

	ret = ov5647_read(sd, 0x4814, &channel_id);
	if (ret < 0)
		return ret;

	channel_id &= ~(3 << 6);
	return ov5647_write(sd, 0x4814, channel_id | (channel << 6));
}

static int ov5647_stream_on(struct v4l2_subdev *sd)
{
	int ret;

	/*
	 * MIPI_CTRL00: clock lane gate + line sync + bus idle = gated
	 * (non-continuous) clock, required to match discontinuous_clk="yes"
	 * in the Tegra device tree mode entries.
	 */
	ret = ov5647_write(sd, 0x4800, 0x34);
	if (ret < 0)
		return ret;

	ret = ov5647_write(sd, 0x4202, 0x00);
	if (ret < 0)
		return ret;

	return ov5647_write(sd, 0x300D, 0x00);
}

static int ov5647_stream_off(struct v4l2_subdev *sd)
{
	int ret;

	/* MIPI_CTRL00: clock lane gate + bus idle + clock lane disable,
	 * parks all MIPI lanes in LP-11 so the CSI receiver sees a clean
	 * LP->HS transition on the next stream on.
	 */
	ret = ov5647_write(sd, 0x4800, 0x25);
	if (ret < 0)
		return ret;

	ret = ov5647_write(sd, 0x4202, 0x0f);
	if (ret < 0)
		return ret;

	return ov5647_write(sd, 0x300D, 0x01);
}

static int set_sw_standby(struct v4l2_subdev *sd, bool standby)
{
	int ret;
	u8 rdval;

	ret = ov5647_read(sd, 0x0100, &rdval);
	if (ret < 0)
		return ret;

	if (standby)
		rdval &= ~0x01;
	else
		rdval |= 0x01;

	return ov5647_write(sd, 0x0100, rdval);
}

static int __sensor_init(struct v4l2_subdev *sd)
{
	int ret;
	u8 resetval, rdval;
	struct i2c_client *client = v4l2_get_subdevdata(sd);

	ret = ov5647_read(sd, 0x0100, &rdval);
	if (ret < 0)
		return ret;

	ret = ov5647_write_array(sd, ov5647_640x480,
					ARRAY_SIZE(ov5647_640x480));
	if (ret < 0) {
		dev_err(&client->dev, "write sensor default regs error\n");
		return ret;
	}

	ret = ov5647_set_virtual_channel(sd, 0);
	if (ret < 0)
		return ret;

	ret = ov5647_read(sd, 0x0100, &resetval);
	if (ret < 0)
		return ret;

	if (!(resetval & 0x01)) {
		dev_err(&client->dev, "Device was in SW standby");
		ret = ov5647_write(sd, 0x0100, 0x01);
		if (ret < 0)
			return ret;
	}

	/* Keep the MIPI lanes in LP-11 until the CSI receiver is armed. */
	return ov5647_stream_off(sd);
}

static int ov5647_sensor_power(struct v4l2_subdev *sd, int on)
{
	int ret = 0;
	struct ov5647 *ov5647 = to_state(sd);
	struct i2c_client *client = v4l2_get_subdevdata(sd);

	mutex_lock(&ov5647->lock);

	if (on && !ov5647->power_count)	{
		dev_dbg(&client->dev, "OV5647 power on\n");

		ret = clk_prepare_enable(ov5647->xclk);
		if (ret < 0) {
			dev_err(&client->dev, "clk prepare enable failed\n");
			goto out;
		}

		ret = ov5647_write_array(sd, sensor_oe_enable_regs,
				ARRAY_SIZE(sensor_oe_enable_regs));
		if (ret < 0) {
			clk_disable_unprepare(ov5647->xclk);
			dev_err(&client->dev,
				"write sensor_oe_enable_regs error\n");
			goto out;
		}

		ret = __sensor_init(sd);
		if (ret < 0) {
			clk_disable_unprepare(ov5647->xclk);
			dev_err(&client->dev,
				"Camera not available, check Power\n");
			goto out;
		}
	} else if (!on && ov5647->power_count == 1) {
		dev_dbg(&client->dev, "OV5647 power off\n");

		ret = ov5647_write_array(sd, sensor_oe_disable_regs,
				ARRAY_SIZE(sensor_oe_disable_regs));

		if (ret < 0)
			dev_dbg(&client->dev, "disable oe failed\n");

		ret = set_sw_standby(sd, true);

		if (ret < 0)
			dev_dbg(&client->dev, "soft stby failed\n");

		clk_disable_unprepare(ov5647->xclk);
	}

	/* Update the power count. */
	ov5647->power_count += on ? 1 : -1;
	WARN_ON(ov5647->power_count < 0);

out:
	mutex_unlock(&ov5647->lock);

	return ret;
}

#ifdef CONFIG_VIDEO_ADV_DEBUG
static int ov5647_sensor_get_register(struct v4l2_subdev *sd,
				struct v4l2_dbg_register *reg)
{
	u8 val;
	int ret;

	ret = ov5647_read(sd, reg->reg & 0xff, &val);
	if (ret < 0)
		return ret;

	reg->val = val;
	reg->size = 1;

	return 0;
}

static int ov5647_sensor_set_register(struct v4l2_subdev *sd,
				const struct v4l2_dbg_register *reg)
{
	return ov5647_write(sd, reg->reg & 0xff, reg->val & 0xff);
}
#endif

/**
 * @short Subdev core operations registration
 */
static const struct v4l2_subdev_core_ops ov5647_subdev_core_ops = {
	.s_power		= ov5647_sensor_power,
#ifdef CONFIG_VIDEO_ADV_DEBUG
	.g_register		= ov5647_sensor_get_register,
	.s_register		= ov5647_sensor_set_register,
#endif
};

static int ov5647_s_stream(struct v4l2_subdev *sd, int enable)
{
	if (enable)
		return ov5647_stream_on(sd);
	else
		return ov5647_stream_off(sd);
}

static const struct v4l2_subdev_video_ops ov5647_subdev_video_ops = {
	.s_stream =		ov5647_s_stream,
};

static int ov5647_enum_mbus_code(struct v4l2_subdev *sd,
				struct v4l2_subdev_pad_config *cfg,
				struct v4l2_subdev_mbus_code_enum *code)
{
	if (code->index > 0)
		return -EINVAL;

	code->code = MEDIA_BUS_FMT_SBGGR10_1X10;

	return 0;
}

static int ov5647_get_fmt(struct v4l2_subdev *sd,
				struct v4l2_subdev_pad_config *cfg,
				struct v4l2_subdev_format *format)
{
	struct v4l2_mbus_framefmt *fmt = &format->format;

	/*
	 * Only one fixed streaming mode is implemented (ov5647_stream_on()
	 * always programs the ov5647_640x480 register table), so report
	 * that resolution here rather than OV5647_WINDOW_*_DEF, which is
	 * the sensor's native full-array size and does not match what is
	 * actually streamed.
	 */
	fmt->code = MEDIA_BUS_FMT_SBGGR10_1X10;
	fmt->width = 640;
	fmt->height = 480;
	fmt->field = V4L2_FIELD_NONE;
	fmt->colorspace = V4L2_COLORSPACE_SRGB;

	return 0;
}

static int ov5647_set_fmt(struct v4l2_subdev *sd,
				struct v4l2_subdev_pad_config *cfg,
				struct v4l2_subdev_format *format)
{
	/* the sensor has a single fixed mode, so set == get */
	return ov5647_get_fmt(sd, cfg, format);
}

static const struct v4l2_subdev_pad_ops ov5647_subdev_pad_ops = {
	.enum_mbus_code = ov5647_enum_mbus_code,
	.get_fmt = ov5647_get_fmt,
	.set_fmt = ov5647_set_fmt,
};

static const struct v4l2_subdev_ops ov5647_subdev_ops = {
	.core		= &ov5647_subdev_core_ops,
	.video		= &ov5647_subdev_video_ops,
	.pad		= &ov5647_subdev_pad_ops,
};

static int ov5647_detect(struct v4l2_subdev *sd)
{
	u8 read;
	int ret;
	struct i2c_client *client = v4l2_get_subdevdata(sd);

	ret = ov5647_write(sd, OV5647_SW_RESET, 0x01);
	if (ret < 0)
		return ret;

	ret = ov5647_read(sd, OV5647_REG_CHIPID_H, &read);
	if (ret < 0)
		return ret;

	if (read != 0x56) {
		dev_err(&client->dev, "ID High expected 0x56 got %x", read);
		return -ENODEV;
	}

	ret = ov5647_read(sd, OV5647_REG_CHIPID_L, &read);
	if (ret < 0)
		return ret;

	if (read != 0x47) {
		dev_err(&client->dev, "ID Low expected 0x47 got %x", read);
		return -ENODEV;
	}

	return ov5647_write(sd, OV5647_SW_RESET, 0x00);
}

static int ov5647_open(struct v4l2_subdev *sd, struct v4l2_subdev_fh *fh)
{
	struct v4l2_mbus_framefmt *format =
				v4l2_subdev_get_try_format(sd, fh->pad, 0);
	struct v4l2_rect *crop =
				v4l2_subdev_get_try_crop(sd, fh->pad, 0);

	crop->left = OV5647_COLUMN_START_DEF;
	crop->top = OV5647_ROW_START_DEF;
	crop->width = OV5647_WINDOW_WIDTH_DEF;
	crop->height = OV5647_WINDOW_HEIGHT_DEF;

	format->code = MEDIA_BUS_FMT_SBGGR10_1X10;

	format->width = OV5647_WINDOW_WIDTH_DEF;
	format->height = OV5647_WINDOW_HEIGHT_DEF;
	format->field = V4L2_FIELD_NONE;
	format->colorspace = V4L2_COLORSPACE_SRGB;

	return 0;
}

static const struct v4l2_subdev_internal_ops ov5647_subdev_internal_ops = {
	.open = ov5647_open,
};

static int ov5647_parse_dt(struct device_node *np)
{
	struct v4l2_of_endpoint bus_cfg;
	struct device_node *ep;

	int ret;

	ep = of_graph_get_next_endpoint(np, NULL);
	if (!ep)
		return -EINVAL;

	ret = v4l2_of_parse_endpoint(ep, &bus_cfg);

	of_node_put(ep);
	return ret;
}

static const struct camera_common_frmfmt ov5647_frmfmt[] = {
	{
		.size = { 640, 480 },
		.framerates = NULL,
		.num_framerates = 0,
		.hdr_en = false,
		.mode = 0,
	},
};

static int ov5647_probe(struct i2c_client *client,
			const struct i2c_device_id *id)
{
	struct device *dev = &client->dev;
	struct ov5647 *sensor;
	struct camera_common_data *s_data;
	int ret;
	struct v4l2_subdev *sd;
	struct device_node *np = client->dev.of_node;
	u32 xclk_freq;

	sensor = devm_kzalloc(dev, sizeof(*sensor), GFP_KERNEL);
	if (!sensor)
		return -ENOMEM;

	s_data = devm_kzalloc(dev, sizeof(*s_data), GFP_KERNEL);
	if (!s_data)
		return -ENOMEM;

	if (IS_ENABLED(CONFIG_OF) && np) {
		ret = ov5647_parse_dt(np);
		if (ret) {
			dev_err(dev, "DT parsing error: %d\n", ret);
			return ret;
		}
	}

	/*
	 * Minimal camera_common integration: this driver keeps its own
	 * fixed-mode get_fmt/set_fmt/s_stream and does not use
	 * camera_common's generic format/control helpers. s_data exists
	 * solely so to_camera_common_data() returns something real (see
	 * the comment on struct ov5647 above), unblocking Tegra's MIPI
	 * PHY calibration for this channel.
	 */
	s_data->dev = dev;
	s_data->priv = sensor;
	s_data->frmfmt = ov5647_frmfmt;
	s_data->numfmts = ARRAY_SIZE(ov5647_frmfmt);
	s_data->def_mode = 0;
	s_data->def_width = 640;
	s_data->def_height = 480;
	s_data->def_clk_freq = 24000000;
	sensor->s_data = s_data;

	ret = camera_common_initialize(s_data, "ov5647");
	if (ret) {
		dev_err(dev, "camera_common_initialize failed: %d\n", ret);
		return ret;
	}
	s_data->mode_prop_idx = 0;

	/*
	 * Get system clock (xclk). On Tegra boards MCLK may be driven
	 * without a DT clocks property, and runs at 24 MHz instead of
	 * the 25 MHz used on the Raspberry Pi: neither is fatal.
	 */
	sensor->xclk = devm_clk_get(dev, NULL);
	if (IS_ERR(sensor->xclk)) {
		dev_warn(dev, "no xclk in DT, assuming MCLK driven externally");
		sensor->xclk = NULL;
	} else {
		xclk_freq = clk_get_rate(sensor->xclk);
		if (xclk_freq != 25000000)
			dev_warn(dev, "unusual xclk frequency: %u, continuing\n",
				 xclk_freq);
	}

	mutex_init(&sensor->lock);

	sd = &s_data->subdev;
	v4l2_i2c_subdev_init(sd, client, &ov5647_subdev_ops);
	sd->internal_ops = &ov5647_subdev_internal_ops;
	sd->flags |= V4L2_SUBDEV_FL_HAS_DEVNODE;

	sensor->pad.flags = MEDIA_PAD_FL_SOURCE;
	sd->entity.function = MEDIA_ENT_F_CAM_SENSOR;
	ret = media_entity_pads_init(&sd->entity, 1, &sensor->pad);
	if (ret < 0)
		goto mutex_remove;

	/* the sensor needs a running MCLK to answer on i2c */
	ret = clk_prepare_enable(sensor->xclk);
	if (ret < 0)
		goto error;
	msleep(20);

	ret = ov5647_detect(sd);
	clk_disable_unprepare(sensor->xclk);
	if (ret < 0)
		goto error;

	ret = v4l2_async_register_subdev(sd);
	if (ret < 0)
		goto error;

	dev_dbg(dev, "OmniVision OV5647 camera driver probed\n");
	return 0;
error:
	media_entity_cleanup(&sd->entity);
mutex_remove:
	mutex_destroy(&sensor->lock);
	return ret;
}

static int ov5647_remove(struct i2c_client *client)
{
	struct v4l2_subdev *sd = i2c_get_clientdata(client);
	struct ov5647 *ov5647 = to_state(sd);

	v4l2_async_unregister_subdev(sd);
	media_entity_cleanup(&sd->entity);
	v4l2_device_unregister_subdev(sd);
	mutex_destroy(&ov5647->lock);

	return 0;
}

static const struct i2c_device_id ov5647_id[] = {
	{ "ov5647", 0 },
	{ }
};
MODULE_DEVICE_TABLE(i2c, ov5647_id);

#if IS_ENABLED(CONFIG_OF)
static const struct of_device_id ov5647_of_match[] = {
	{ .compatible = "ovti,ov5647" },
	{ /* sentinel */ },
};
MODULE_DEVICE_TABLE(of, ov5647_of_match);
#endif

static struct i2c_driver ov5647_driver = {
	.driver = {
		.of_match_table = of_match_ptr(ov5647_of_match),
		.name	= SENSOR_NAME,
	},
	.probe		= ov5647_probe,
	.remove		= ov5647_remove,
	.id_table	= ov5647_id,
};

module_i2c_driver(ov5647_driver);

MODULE_AUTHOR("Ramiro Oliveira <roliveir@synopsys.com>");
MODULE_DESCRIPTION("A low-level driver for OmniVision ov5647 sensors");
MODULE_LICENSE("GPL v2");
