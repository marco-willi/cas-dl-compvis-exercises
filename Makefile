#################################################################################
# GLOBALS                                                                       #
#################################################################################

ifneq (,$(wildcard ./.env))
    include .env
    export
endif

NOTEBOOK_PORT := 8888
NOTEBOOK_DIR  := .

#################################################################################
# COMMANDS                                                                       #
#################################################################################

.PHONY: help
help: ## Show all available commands
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

##### JUPYTER

.PHONY: notebook
notebook: ## Start Jupyter notebook server (PORT=8888, DIR=.)
	jupyter notebook --port=$(NOTEBOOK_PORT) --notebook-dir=$(NOTEBOOK_DIR) --no-browser --allow-root
