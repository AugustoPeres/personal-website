+++
title = "Augusto's emacs config"
author = ["Augusto Peres"]
draft = false
+++

## Package management {#package-management}

This section was taken from [Distro Tubes video](https://www.youtube.com/watch?v=hoP4I9ounyQ&t=504s&ab_channel=DistroTube)


### Setup package.el to work with MELPA {#setup-package-dot-el-to-work-with-melpa}

```emacs-lisp
(require 'package)
(add-to-list 'package-archives
			    '("melpa" . "https://melpa.org/packages/"))
(package-refresh-contents)
(package-initialize)
```


### Instaling use_package {#instaling-use-package}

```emacs-lisp
(unless (package-installed-p 'use-package)
  (package-install 'use-package))
(setq use-package-always-ensure t)
```


## GUI tweeks {#gui-tweeks}


### <span class="org-todo todo TODO">TODO</span> Fonts {#fonts}


### Disable menu bar {#disable-menu-bar}

This section was taken from [Distro Tubes video](https://www.youtube.com/watch?v=hoP4I9ounyQ&t=504s&ab_channel=DistroTube) aprt from indicated changes.

For some reason this code is not working on MacIOs terminal

```elisp
(menu-bar-mode -1)
(tool-bar-mode -1)
(scroll-bar-mode -1)
```


### Display line numbers and truncated lines {#display-line-numbers-and-truncated-lines}

This section was taken from [Distro Tubes video](https://www.youtube.com/watch?v=hoP4I9ounyQ&t=504s&ab_channel=DistroTube) aprt from indicated changes.
The relative line numbers part was taken from this [this stack exchange](https://stackoverflow.com/questions/6874516/relative-line-numbers-in-emacs)

```emacs-lisp
(global-display-line-numbers-mode 1)
(global-visual-line-mode t)
(setq display-line-numbers-type 'relative)
```


### Change to Doom's modeline {#change-to-doom-s-modeline}

This section was taken from [Distro Tubes video](https://www.youtube.com/watch?v=hoP4I9ounyQ&t=504s&ab_channel=DistroTube) aprt from indicated changes.
For some reason Distro Tube's code was not working on MacIOs terminal
Changed here to follow the [official docs](https://github.com/seagle0128/doom-modeline) with better results

```emacs-lisp
;; (use-package doom-modeline)
;; (doom-modeline-mode 1)
(use-package doom-modeline
  :ensure t
  :hook (after-init . doom-modeline-mode))
```


## Evil mode {#evil-mode}

This section was taken from [Distro Tubes video](https://www.youtube.com/watch?v=hoP4I9ounyQ&t=504s&ab_channel=DistroTube)

```elisp
(use-package evil
  :init      ;; tweak evil's configuration before loading it
  (setq evil-want-integration t) ;; This is optional since it's already set to t by default.
  (setq evil-want-keybinding nil)
  (setq evil-vsplit-window-right t)
  (setq evil-split-window-below t)
  (evil-mode))
(use-package evil-collection
  :after evil
  :config
  (setq evil-collection-mode-list '(dashboard dired ibuffer))
  (evil-collection-init))
(use-package evil-tutor)
```


## General key bindings {#general-key-bindings}

This section was taken from [Distro Tubes video](https://www.youtube.com/watch?v=hoP4I9ounyQ&t=504s&ab_channel=DistroTube)

```elisp
(use-package general
  :config
  (general-evil-setup t))
```


## Using doom themes {#using-doom-themes}

This was taken directly from the package [github](https://github.com/doomemacs/themes)

```elisp
(use-package doom-themes
  :ensure t
  :config
  ;; Global settings (defaults)
  (setq doom-themes-enable-bold t    ; if nil, bold is universally disabled
	  doom-themes-enable-italic t) ; if nil, italics is universally disabled
  (load-theme 'doom-one t)

  ;; Enable flashing mode-line on errors
  (doom-themes-visual-bell-config)
  ;; Enable custom neotree theme (all-the-icons must be installed!)
  (doom-themes-neotree-config)
  ;; or for treemacs users
  (setq doom-themes-treemacs-theme "doom-atom") ; use "doom-colors" for less minimal icon theme
  (doom-themes-treemacs-config)
  ;; Corrects (and improves) org-mode's native fontification.
  (doom-themes-org-config))
```


## Key bindings setup with general {#key-bindings-setup-with-general}


## Automatically close and open brackets {#automatically-close-and-open-brackets}

```elisp
(electric-pair-mode t)
```


## Install which-key {#install-which-key}

Install this package using melpa `M-x package-install which-key`

```elisp
(which-key-mode)
```


## <span class="org-todo todo TODO">TODO</span> Install an autocomplete {#install-an-autocomplete}

Installed [company mode](http://company-mode.github.io/) with `M-x package-install company`.

To disable autocomplete on enter I followed [this comment](https://www.reddit.com/r/emacs/comments/q8u2l4/unsetting_return_in_company_mode/) on reddit.

```elisp
;; This enables it on all buffers.
(add-hook 'after-init-hook 'global-company-mode)
(use-package company
  :bind(:map company-active-map
         ("<return>" . nil)
         ("RET" . nil)
         ("<tab>" . company-complete-selection)))
```


## Handling meta keys in MacOs {#handling-meta-keys-in-macos}

```elisp
(setq mac-option-modifier nil
      mac-command-modifier 'meta
      x-select-enable-clipboard t)
```


## Front page (dashboard) {#front-page--dashboard}

This uses the [dashboard](https://github.com/emacs-dashboard/emacs-dashboard) package.

```elisp
(use-package dashboard
    :ensure t
    :config
    (dashboard-setup-startup-hook))

(setq dashboard-banner-logo-title "Welcome EMACS. Reject VsCode")
(setq dashboard-items '((recents  . 10)))
```


## <span class="org-todo todo TODO">TODO</span> Install a spell checker {#install-a-spell-checker}


## <span class="org-todo todo TODO">TODO</span> Install Magit {#install-magit}

Install from MELPA `M-x paclage-install RET magit`. After this look at the [post install instructions](https://magit.vc/manual/magit/Post_002dInstallation-Tasks.html)


## Org mode configurations {#org-mode-configurations}


### Latex {#latex}

```elisp
(plist-put org-format-latex-options :scale 1.5)
```


### General org mode stuff {#general-org-mode-stuff}

This makes sure that a new line is created when we get to 80 characters:

```elisp
(add-hook 'org-mode-hook '(lambda () (setq fill-column 80)))
(add-hook 'org-mode-hook 'turn-on-auto-fill)
```


### Adding python to exectution code blocks {#adding-python-to-exectution-code-blocks}

```elisp
(org-babel-do-load-languages
 'org-babel-load-languages
 '((python . t)
   (shell . t)))
```


### No identation in code-blocks {#no-identation-in-code-blocks}

```elisp
(setq org-src-preserve-indentation t)
```


### Setting up indentation as default {#setting-up-indentation-as-default}

```elisp
(add-hook 'org-mode-hook 'org-indent-mode)
```


### Setting up org-capture {#setting-up-org-capture}

```elisp
(setq org-capture-templates
        '(("j" "Journal" entry (file+datetree "~/Documents/Org/journal-2023.org")
           "* %?\nEntered on %U\n" :time-prompt 1)))

(global-set-key (kbd "C-c o c") 'org-capture)
```


### Exporting to Markdown {#exporting-to-markdown}

First: `M-x package-install RET ox-gfm`. Then load the package automatically.

```elisp
(eval-after-load "org"
  '(require 'ox-gfm nil t))
```


### Setting up org-roam {#setting-up-org-roam}


#### <span class="org-todo todo TODO">TODO</span> Use org files as templates {#use-org-files-as-templates}


#### <span class="org-todo todo TODO">TODO</span> See if I can use templates to write to an existing file {#see-if-i-can-use-templates-to-write-to-an-existing-file}

```elisp
(use-package org-roam
    :ensure t)

(use-package org-roam
  :ensure t
  :custom
  (org-roam-directory "~/Documents/org-roam-brain")
  (org-roam-capture-templates
   '(("b" "book" plain
       "\n* ${title}\n - Author: %^{Author}\n - Year: %^{Year}\n* Summary\n\n%?\n[[id:0ef132d8-6cf7-4228-a0f1-98d0dcb72f8a][BOOKS]]"
      :if-new (file+head "%<%Y%m%d%H%M%S>-${slug}.org" "#+title: ${title}\n")
      :unnarrowed t)
     ("d" "default" plain
      "%?"
      :if-new (file+head "%<%Y%m%d%H%M%S>-${slug}.org" "#+title: ${title}\n")
      :unnarrowed t)
     ("i" "ingredient" plain "* ${ingredient name}\n:PROPERTIES:\n:ID: %(org-id-uuid)\n:END:\n%?"
      :target (file "ingredients.org"))
     ("r" "recipe" plain
      "\n* ${title}\n - Tempo preparaçao: %^{tempo preparacao (min)} min\n - Serve %^{ serve quantas pessoas} pessoas\n* Ingredients\n%? \n* Preparation\n[[id:42f44c53-e440-4f96-9f79-29d6718c51d5][RECIPE]]"
      :if-new (file+head "%<%Y%m%d%H%M%S>-recipe-${slug}.org" "#+title: ${title}\n")
      :unnarrowed t)
))
  :bind (("C-c n l" . org-roam-buffer-toggle)
         ("C-c n f" . org-roam-node-find)
         ("C-c n i" . org-roam-node-insert)
         ("C-c n c" . org-roam-capture))
  :config
  (org-roam-setup))
```

To make each PC build the database we can use: `org-roam-db-autosync-mode`.


## Setting up `ox-hugo` {#setting-up-ox-hugo}

[Ox-hugo](https://ox-hugo.scripter.co/) allows easy export from org-mode to hugo markdown.

```elisp
(use-package ox-hugo
  :ensure t   ;Auto-install the package from Melpa
  :pin melpa
  :after ox)
```

Note. Before exporting we must set the `#+hugo_base_dir:<path/to/contents>` in
an org file. See the example at the beggining of this file.


## Ivy completion framework {#ivy-completion-framework}

Installed the package with `M-x package-install RET ivy`.

By default this already gives me the spacemacs behaviour on finding
and navigating files when using the default key bindings:

-   `C-x C-f` opens a menu that allows me to search for files;
-   `C-x b` opens the menu buffer with the option to search the files.

To enable `ivy-mode` by default we simply need to add:

```elisp
(ivy-mode)
```


## Setting up things for python {#setting-up-things-for-python}


### Anaconda mode {#anaconda-mode}

Use `M-x package-install RET anaconda mode`

```elisp
(add-hook 'python-mode-hook 'anaconda-mode)
```

To get suggestion on what the function arguments should be we should use eldoc.

```elisp
(add-hook 'python-mode-hook 'anaconda-eldoc-mode)
```

Then I installed [company anaconda](https://github.com/pythonic-emacs/company-anaconda) to use company mode and have anaconda-mode
with a better user interface.

```elisp
(eval-after-load "company"
 '(add-to-list 'company-backends 'company-anaconda))
```


### Yapf formatting {#yapf-formatting}

Installed with `M-x package-install -RET yapfify`.

```elisp
(add-hook 'python-mode-hook 'yapf-mode)
```

In theory this should automatically yapify a buffer on save. But, if that does
not happen we can call `M-x yapfify-buffer`.


## Setting up copilot {#setting-up-copilot}

Following the instructions [here](https://github.com/zerolfx/copilot.el). But first had to make the next setup
configuration on `quelpa` to get it working with the syntax in the second
codeblock. This setup was thaken from [here](https://github.com/quelpa/quelpa-use-package).

```elisp
(quelpa
 '(quelpa-use-package
   :fetcher git
   :url "https://github.com/quelpa/quelpa-use-package.git"))
(require 'quelpa-use-package)
```

```elisp
(use-package copilot

  :quelpa (copilot :fetcher github
                                      :repo "zerolfx/copilot.el"
                                      :branch "main"
                                      :files ("dist" "*.el"))
  :bind (:map copilot-completion-map
              ("<tab>" . 'copilot-accept-completion)
              ("TAB" . 'copilot-accept-completion)
              ("C-TAB" . 'copilot-accept-completion-by-word)
              ("C-<tab>" . 'copilot-accept-completion-by-word))
)
```


## Distraction free mode {#distraction-free-mode}

To have a distraction free mode I installed [writeroom-mode](https://github.com/joostkremers/writeroom-mode) directly using Melpa.

```elisp
(add-hook 'writeroom-mode-hook (lambda () (display-line-numbers-mode -1)))
```


## Yasnippets {#yasnippets}

Install [yasnippet](https://github.com/joaotavora/yasnippet) from MELPA with `M-x package-install RET yasnippet`.

```elisp
(setq yas-snippet-dirs '("~/emas-config/yas-snippets"))
(yas-global-mode t)
```

At the moment I was not able to add `yasnippet` to use `M-x company-yasnippet`
to get a dropdown menu of possible completions.
