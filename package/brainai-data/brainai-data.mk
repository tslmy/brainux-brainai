################################################################################
#
# brainai-data
#
################################################################################

# Hugging Face revision of yamada2wasabi/brain-ai-v1-preview.
BRAINAI_DATA_VERSION = 19f84a22ebd742cbde5c90095e1f76ebffff295a
BRAINAI_DATA_SITE = https://huggingface.co/yamada2wasabi/brain-ai-v1-preview/resolve/$(BRAINAI_DATA_VERSION)
BRAINAI_DATA_SOURCE = model.safetensors
BRAINAI_DATA_UNIFONT_VERSION = 17.0.04
BRAINAI_DATA_UNIFONT = unifont_jp-$(BRAINAI_DATA_UNIFONT_VERSION).bdf.gz
BRAINAI_DATA_EXTRA_DOWNLOADS = \
	$(BRAINAI_DATA_SITE)/tokenizer.model \
	$(BRAINAI_DATA_SITE)/config.json \
	https://ftp.gnu.org/gnu/unifont/unifont-$(BRAINAI_DATA_UNIFONT_VERSION)/$(BRAINAI_DATA_UNIFONT)
BRAINAI_DATA_LICENSE = CC-BY-SA-4.0 (model), OFL-1.1 or GPL-2.0+ with font embedding exception (Unifont)
BRAINAI_DATA_REDISTRIBUTE = YES

BRAINAI_DATA_TOOLS = $(BR2_EXTERNAL_BRAINAI_PATH)/package/brainai/src/tools
BRAINAI_DATA_SRC = $(BR2_EXTERNAL_BRAINAI_PATH)/package/brainai/src

# The downloads are plain files, not archives.
define BRAINAI_DATA_EXTRACT_CMDS
	cp $(BRAINAI_DATA_DL_DIR)/model.safetensors $(BRAINAI_DATA_DL_DIR)/tokenizer.model \
		$(BRAINAI_DATA_DL_DIR)/config.json $(BRAINAI_DATA_DL_DIR)/$(BRAINAI_DATA_UNIFONT) $(@D)/
endef

# Standard-library-only Python: pack the ternary weights (~6.7 MB) and
# subset the font to the characters brainai can ever print.
define BRAINAI_DATA_BUILD_CMDS
	python3 $(BRAINAI_DATA_TOOLS)/convert.py $(@D)/model.safetensors \
		$(@D)/tokenizer.model $(@D)/out
	python3 $(BRAINAI_DATA_TOOLS)/mkfont.py $(@D)/$(BRAINAI_DATA_UNIFONT) \
		$(@D)/out/tok.bin $(@D)/out/font.bin \
		$(BRAINAI_DATA_SRC)/main.c $(BRAINAI_DATA_SRC)/text.c $(BRAINAI_DATA_SRC)/ui_fb.c
endef

define BRAINAI_DATA_INSTALL_TARGET_CMDS
	$(INSTALL) -D -m 0644 $(@D)/out/model.bin $(TARGET_DIR)/usr/share/brainai/model.bin
	$(INSTALL) -D -m 0644 $(@D)/out/tok.bin $(TARGET_DIR)/usr/share/brainai/tok.bin
	$(INSTALL) -D -m 0644 $(@D)/out/font.bin $(TARGET_DIR)/usr/share/brainai/font.bin
endef

$(eval $(generic-package))
