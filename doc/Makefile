# Minimal makefile for Sphinx documentation

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build
DEPLOYDIR = ../docs

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

html:
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

deploy:
	@rm -rf $(DEPLOYDIR)
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	@mkdir $(DEPLOYDIR)
	@cp -r $(BUILDDIR)/html/* $(DEPLOYDIR)
	@touch $(DEPLOYDIR)/.nojekyll
	@echo "\nBuilt documentation under '$(DEPLOYDIR)'."

clean:
	@rm -rf $(BUILDDIR)
	@rm -rf auto_examples
	@rm -rf generated
	@rm -rf _autosummary
	@echo "'$(BUILDDIR)', 'auto_examples', 'generated' directories deleted."

clean-deploy:
	@rm -rf $(DEPLOYDIR)
	@echo "'$(DEPLOYDIR)' directory deleted."

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help html deploy clean clean-deploy Makefile
