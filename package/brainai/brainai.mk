################################################################################
#
# brainai
#
################################################################################

BRAINAI_VERSION = 1.0.0
BRAINAI_SITE = $(BR2_EXTERNAL_BRAINAI_PATH)/package/brainai/src
BRAINAI_SITE_METHOD = local
BRAINAI_LICENSE = MIT
BRAINAI_LICENSE_FILES = LICENSE

# The ternary/LUT kernels are the hot path; build them at -O2 even when the
# rest of the system is optimised for size.
define BRAINAI_BUILD_CMDS
	$(MAKE) $(TARGET_CONFIGURE_OPTS) CFLAGS="$(TARGET_CFLAGS) -O2" \
		DATADIR=/usr/share/brainai -C $(@D)
endef

define BRAINAI_INSTALL_TARGET_CMDS
	$(INSTALL) -D -m 0755 $(@D)/brainai $(TARGET_DIR)/usr/bin/brainai
endef

$(eval $(generic-package))
