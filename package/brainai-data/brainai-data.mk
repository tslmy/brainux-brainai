################################################################################
#
# brainai-data
#
################################################################################

# Hugging Face revision of yamada2wasabi/brain-ai-v1-preview.
BRAINAI_DATA_VERSION = 19f84a22ebd742cbde5c90095e1f76ebffff295a
BRAINAI_DATA_SITE = https://huggingface.co/yamada2wasabi/brain-ai-v1-preview/resolve/$(BRAINAI_DATA_VERSION)
BRAINAI_DATA_SOURCE = model.safetensors
BRAINAI_DATA_EXTRA_DOWNLOADS = \
	$(BRAINAI_DATA_SITE)/tokenizer.model \
	$(BRAINAI_DATA_SITE)/config.json
BRAINAI_DATA_LICENSE = CC-BY-SA-4.0
BRAINAI_DATA_REDISTRIBUTE = YES

BRAINAI_DATA_TOOLS = $(BR2_EXTERNAL_BRAINAI_PATH)/package/brainai/src/tools

# The downloads are plain files, not archives.
define BRAINAI_DATA_EXTRACT_CMDS
	cp $(BRAINAI_DATA_DL_DIR)/model.safetensors $(BRAINAI_DATA_DL_DIR)/tokenizer.model \
		$(BRAINAI_DATA_DL_DIR)/config.json $(@D)/
endef

# Standard-library-only Python: pack the ternary weights (~6.7 MB) and
# the tokenizer vocabulary.
define BRAINAI_DATA_BUILD_CMDS
	python3 $(BRAINAI_DATA_TOOLS)/convert.py $(@D)/model.safetensors \
		$(@D)/tokenizer.model $(@D)/out
endef

define BRAINAI_DATA_INSTALL_TARGET_CMDS
	$(INSTALL) -D -m 0644 $(@D)/out/model.bin $(TARGET_DIR)/usr/share/brainai/model.bin
	$(INSTALL) -D -m 0644 $(@D)/out/tok.bin $(TARGET_DIR)/usr/share/brainai/tok.bin
endef

$(eval $(generic-package))
