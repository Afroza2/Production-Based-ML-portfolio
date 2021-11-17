# Production-Based-ML-portfolio

I created a basic flask-based portal where a person can give input on 14 parameters. I integrated a classification model to this project which predicts on these input data and lets the person knows if there is a risk if heart disease or not. 


The dataset is taken from - https://www.kaggle.com/ronitf/heart-disease-uci


## Installation

Install with pip:

```
$ pip install -r requirements.txt
```

## Flask Application Structure 
```
.
.
├── app.py
├── dataset
│   └── HeartDiseaseDataset.csv
├── JupyterNotebook
│   ├── Heart Disease Dataset.csv
│   └── HeartDiseases.ipynb
├── __pycache__
│   ├── app.cpython-36.pyc
│   └── app.cpython-39.pyc
├── requirements.txt
├── resource
│   └── diseaseprediction.joblib
├── static
├── templates
│   └── home.html
└── venv
    ├── bin
    │   ├── activate
    │   ├── activate.csh
    │   ├── activate.fish
    │   ├── activate.ps1
    │   ├── activate_this.py
    │   ├── activate.xsh
    │   ├── flask
    │   ├── pip
    │   ├── pip3
    │   ├── pip-3.9
    │   ├── pip3.9
    │   ├── python -> /home/senjuti/.pyenv/versions/3.9.0/bin/python
    │   ├── python3 -> python
    │   ├── python3.9 -> python
    │   ├── wheel
    │   ├── wheel3
    │   ├── wheel-3.9
    │   └── wheel3.9
    ├── lib
    │   └── python3.9
    │       └── site-packages
    │           ├── click
    │           │   ├── _compat.py
    │           │   ├── core.py
    │           │   ├── decorators.py
    │           │   ├── exceptions.py
    │           │   ├── formatting.py
    │           │   ├── globals.py
    │           │   ├── __init__.py
    │           │   ├── parser.py
    │           │   ├── __pycache__
    │           │   │   ├── _compat.cpython-39.pyc
    │           │   │   ├── core.cpython-39.pyc
    │           │   │   ├── decorators.cpython-39.pyc
    │           │   │   ├── exceptions.cpython-39.pyc
    │           │   │   ├── formatting.cpython-39.pyc
    │           │   │   ├── globals.cpython-39.pyc
    │           │   │   ├── __init__.cpython-39.pyc
    │           │   │   ├── parser.cpython-39.pyc
    │           │   │   ├── shell_completion.cpython-39.pyc
    │           │   │   ├── termui.cpython-39.pyc
    │           │   │   ├── _termui_impl.cpython-39.pyc
    │           │   │   ├── testing.cpython-39.pyc
    │           │   │   ├── _textwrap.cpython-39.pyc
    │           │   │   ├── types.cpython-39.pyc
    │           │   │   ├── _unicodefun.cpython-39.pyc
    │           │   │   ├── utils.cpython-39.pyc
    │           │   │   └── _winconsole.cpython-39.pyc
    │           │   ├── py.typed
    │           │   ├── shell_completion.py
    │           │   ├── _termui_impl.py
    │           │   ├── termui.py
    │           │   ├── testing.py
    │           │   ├── _textwrap.py
    │           │   ├── types.py
    │           │   ├── _unicodefun.py
    │           │   ├── utils.py
    │           │   └── _winconsole.py
    │           ├── click-8.0.3.dist-info
    │           │   ├── INSTALLER
    │           │   ├── LICENSE.rst
    │           │   ├── METADATA
    │           │   ├── RECORD
    │           │   ├── top_level.txt
    │           │   └── WHEEL
    │           ├── _distutils_hack
    │           │   ├── __init__.py
    │           │   └── override.py
    │           ├── distutils-precedence.pth
    │           ├── flask
    │           │   ├── app.py
    │           │   ├── blueprints.py
    │           │   ├── cli.py
    │           │   ├── config.py
    │           │   ├── ctx.py
    │           │   ├── debughelpers.py
    │           │   ├── globals.py
    │           │   ├── helpers.py
    │           │   ├── __init__.py
    │           │   ├── json
    │           │   │   ├── __init__.py
    │           │   │   ├── __pycache__
    │           │   │   │   ├── __init__.cpython-39.pyc
    │           │   │   │   └── tag.cpython-39.pyc
    │           │   │   └── tag.py
    │           │   ├── logging.py
    │           │   ├── __main__.py
    │           │   ├── __pycache__
    │           │   │   ├── app.cpython-39.pyc
    │           │   │   ├── blueprints.cpython-39.pyc
    │           │   │   ├── cli.cpython-39.pyc
    │           │   │   ├── config.cpython-39.pyc
    │           │   │   ├── ctx.cpython-39.pyc
    │           │   │   ├── debughelpers.cpython-39.pyc
    │           │   │   ├── globals.cpython-39.pyc
    │           │   │   ├── helpers.cpython-39.pyc
    │           │   │   ├── __init__.cpython-39.pyc
    │           │   │   ├── logging.cpython-39.pyc
    │           │   │   ├── __main__.cpython-39.pyc
    │           │   │   ├── scaffold.cpython-39.pyc
    │           │   │   ├── sessions.cpython-39.pyc
    │           │   │   ├── signals.cpython-39.pyc
    │           │   │   ├── templating.cpython-39.pyc
    │           │   │   ├── testing.cpython-39.pyc
    │           │   │   ├── typing.cpython-39.pyc
    │           │   │   ├── views.cpython-39.pyc
    │           │   │   └── wrappers.cpython-39.pyc
    │           │   ├── py.typed
    │           │   ├── scaffold.py
    │           │   ├── sessions.py
    │           │   ├── signals.py
    │           │   ├── templating.py
    │           │   ├── testing.py
    │           │   ├── typing.py
    │           │   ├── views.py
    │           │   └── wrappers.py
    │           ├── Flask-2.0.2.dist-info
    │           │   ├── entry_points.txt
    │           │   ├── INSTALLER
    │           │   ├── LICENSE.rst
    │           │   ├── METADATA
    │           │   ├── RECORD
    │           │   ├── REQUESTED
    │           │   ├── top_level.txt
    │           │   └── WHEEL
    │           ├── itsdangerous
    │           │   ├── encoding.py
    │           │   ├── exc.py
    │           │   ├── __init__.py
    │           │   ├── _json.py
    │           │   ├── jws.py
    │           │   ├── __pycache__
    │           │   │   ├── encoding.cpython-39.pyc
    │           │   │   ├── exc.cpython-39.pyc
    │           │   │   ├── __init__.cpython-39.pyc
    │           │   │   ├── _json.cpython-39.pyc
    │           │   │   ├── jws.cpython-39.pyc
    │           │   │   ├── serializer.cpython-39.pyc
    │           │   │   ├── signer.cpython-39.pyc
    │           │   │   ├── timed.cpython-39.pyc
    │           │   │   └── url_safe.cpython-39.pyc
    │           │   ├── py.typed
    │           │   ├── serializer.py
    │           │   ├── signer.py
    │           │   ├── timed.py
    │           │   └── url_safe.py
    │           ├── itsdangerous-2.0.1.dist-info
    │           │   ├── INSTALLER
    │           │   ├── LICENSE.rst
    │           │   ├── METADATA
    │           │   ├── RECORD
    │           │   ├── top_level.txt
    │           │   └── WHEEL
    │           ├── jinja2
    │           │   ├── async_utils.py
    │           │   ├── bccache.py
    │           │   ├── compiler.py
    │           │   ├── constants.py
    │           │   ├── debug.py
    │           │   ├── defaults.py
    │           │   ├── environment.py
    │           │   ├── exceptions.py
    │           │   ├── ext.py
    │           │   ├── filters.py
    │           │   ├── _identifier.py
    │           │   ├── idtracking.py
    │           │   ├── __init__.py
    │           │   ├── lexer.py
    │           │   ├── loaders.py
    │           │   ├── meta.py
    │           │   ├── nativetypes.py
    │           │   ├── nodes.py
    │           │   ├── optimizer.py
    │           │   ├── parser.py
    │           │   ├── __pycache__
    │           │   │   ├── async_utils.cpython-39.pyc
    │           │   │   ├── bccache.cpython-39.pyc
    │           │   │   ├── compiler.cpython-39.pyc
    │           │   │   ├── constants.cpython-39.pyc
    │           │   │   ├── debug.cpython-39.pyc
    │           │   │   ├── defaults.cpython-39.pyc
    │           │   │   ├── environment.cpython-39.pyc
    │           │   │   ├── exceptions.cpython-39.pyc
    │           │   │   ├── ext.cpython-39.pyc
    │           │   │   ├── filters.cpython-39.pyc
    │           │   │   ├── _identifier.cpython-39.pyc
    │           │   │   ├── idtracking.cpython-39.pyc
    │           │   │   ├── __init__.cpython-39.pyc
    │           │   │   ├── lexer.cpython-39.pyc
    │           │   │   ├── loaders.cpython-39.pyc
    │           │   │   ├── meta.cpython-39.pyc
    │           │   │   ├── nativetypes.cpython-39.pyc
    │           │   │   ├── nodes.cpython-39.pyc
    │           │   │   ├── optimizer.cpython-39.pyc
    │           │   │   ├── parser.cpython-39.pyc
    │           │   │   ├── runtime.cpython-39.pyc
    │           │   │   ├── sandbox.cpython-39.pyc
    │           │   │   ├── tests.cpython-39.pyc
    │           │   │   ├── utils.cpython-39.pyc
    │           │   │   └── visitor.cpython-39.pyc
    │           │   ├── py.typed
    │           │   ├── runtime.py
    │           │   ├── sandbox.py
    │           │   ├── tests.py
    │           │   ├── utils.py
    │           │   └── visitor.py
    │           ├── Jinja2-3.0.3.dist-info
    │           │   ├── entry_points.txt
    │           │   ├── INSTALLER
    │           │   ├── LICENSE.rst
    │           │   ├── METADATA
    │           │   ├── RECORD
    │           │   ├── top_level.txt
    │           │   └── WHEEL
    │           ├── markupsafe
    │           │   ├── __init__.py
    │           │   ├── _native.py
    │           │   ├── __pycache__
    │           │   │   ├── __init__.cpython-39.pyc
    │           │   │   └── _native.cpython-39.pyc
    │           │   ├── py.typed
    │           │   ├── _speedups.c
    │           │   ├── _speedups.cpython-39-x86_64-linux-gnu.so
    │           │   └── _speedups.pyi
    │           ├── MarkupSafe-2.0.1.dist-info
    │           │   ├── INSTALLER
    │           │   ├── LICENSE.rst
    │           │   ├── METADATA
    │           │   ├── RECORD
    │           │   ├── top_level.txt
    │           │   └── WHEEL
    │           ├── pip
    │           │   ├── __init__.py
    │           │   ├── _internal
    │           │   │   ├── build_env.py
    │           │   │   ├── cache.py
    │           │   │   ├── cli
    │           │   │   │   ├── autocompletion.py
    │           │   │   │   ├── base_command.py
    │           │   │   │   ├── cmdoptions.py
    │           │   │   │   ├── command_context.py
    │           │   │   │   ├── __init__.py
    │           │   │   │   ├── main_parser.py
    │           │   │   │   ├── main.py
    │           │   │   │   ├── parser.py
    │           │   │   │   ├── progress_bars.py
    │           │   │   │   ├── req_command.py
    │           │   │   │   ├── spinners.py
    │           │   │   │   └── status_codes.py
    │           │   │   ├── commands
    │           │   │   │   ├── cache.py
    │           │   │   │   ├── check.py
    │           │   │   │   ├── completion.py
    │           │   │   │   ├── configuration.py
    │           │   │   │   ├── debug.py
    │           │   │   │   ├── download.py
    │           │   │   │   ├── freeze.py
    │           │   │   │   ├── hash.py
    │           │   │   │   ├── help.py
    │           │   │   │   ├── __init__.py
    │           │   │   │   ├── install.py
    │           │   │   │   ├── list.py
    │           │   │   │   ├── search.py
    │           │   │   │   ├── show.py
    │           │   │   │   ├── uninstall.py
    │           │   │   │   └── wheel.py
    │           │   │   ├── configuration.py
    │           │   │   ├── distributions
    │           │   │   │   ├── base.py
    │           │   │   │   ├── __init__.py
    │           │   │   │   ├── installed.py
    │           │   │   │   ├── sdist.py
    │           │   │   │   └── wheel.py
    │           │   │   ├── exceptions.py
    │           │   │   ├── index
    │           │   │   │   ├── collector.py
    │           │   │   │   ├── __init__.py
    │           │   │   │   ├── package_finder.py
    │           │   │   │   └── sources.py
    │           │   │   ├── __init__.py
    │           │   │   ├── locations
    │           │   │   │   ├── base.py
    │           │   │   │   ├── _distutils.py
    │           │   │   │   ├── __init__.py
    │           │   │   │   └── _sysconfig.py
    │           │   │   ├── main.py
    │           │   │   ├── metadata
    │           │   │   │   ├── base.py
    │           │   │   │   ├── __init__.py
    │           │   │   │   └── pkg_resources.py
    │           │   │   ├── models
    │           │   │   │   ├── candidate.py
    │           │   │   │   ├── direct_url.py
    │           │   │   │   ├── format_control.py
    │           │   │   │   ├── index.py
    │           │   │   │   ├── __init__.py
    │           │   │   │   ├── link.py
    │           │   │   │   ├── scheme.py
    │           │   │   │   ├── search_scope.py
    │           │   │   │   ├── selection_prefs.py
    │           │   │   │   ├── target_python.py
    │           │   │   │   └── wheel.py
    │           │   │   ├── network
    │           │   │   │   ├── auth.py
    │           │   │   │   ├── cache.py
    │           │   │   │   ├── download.py
    │           │   │   │   ├── __init__.py
    │           │   │   │   ├── lazy_wheel.py
    │           │   │   │   ├── session.py
    │           │   │   │   ├── utils.py
    │           │   │   │   └── xmlrpc.py
    │           │   │   ├── operations
    │           │   │   │   ├── build
    │           │   │   │   │   ├── __init__.py
    │           │   │   │   │   ├── metadata_legacy.py
    │           │   │   │   │   ├── metadata.py
    │           │   │   │   │   ├── wheel_legacy.py
    │           │   │   │   │   └── wheel.py
    │           │   │   │   ├── check.py
    │           │   │   │   ├── freeze.py
    │           │   │   │   ├── __init__.py
    │           │   │   │   ├── install
    │           │   │   │   │   ├── editable_legacy.py
    │           │   │   │   │   ├── __init__.py
    │           │   │   │   │   ├── legacy.py
    │           │   │   │   │   └── wheel.py
    │           │   │   │   └── prepare.py
    │           │   │   ├── pyproject.py
    │           │   │   ├── req
    │           │   │   │   ├── constructors.py
    │           │   │   │   ├── __init__.py
    │           │   │   │   ├── req_file.py
    │           │   │   │   ├── req_install.py
    │           │   │   │   ├── req_set.py
    │           │   │   │   ├── req_tracker.py
    │           │   │   │   └── req_uninstall.py
    │           │   │   ├── resolution
    │           │   │   │   ├── base.py
    │           │   │   │   ├── __init__.py
    │           │   │   │   ├── legacy
    │           │   │   │   │   ├── __init__.py
    │           │   │   │   │   └── resolver.py
    │           │   │   │   └── resolvelib
    │           │   │   │       ├── base.py
    │           │   │   │       ├── candidates.py
    │           │   │   │       ├── factory.py
    │           │   │   │       ├── found_candidates.py
    │           │   │   │       ├── __init__.py
    │           │   │   │       ├── provider.py
    │           │   │   │       ├── reporter.py
    │           │   │   │       ├── requirements.py
    │           │   │   │       └── resolver.py
    │           │   │   ├── self_outdated_check.py
    │           │   │   ├── utils
    │           │   │   │   ├── appdirs.py
    │           │   │   │   ├── compatibility_tags.py
    │           │   │   │   ├── compat.py
    │           │   │   │   ├── datetime.py
    │           │   │   │   ├── deprecation.py
    │           │   │   │   ├── direct_url_helpers.py
    │           │   │   │   ├── distutils_args.py
    │           │   │   │   ├── encoding.py
    │           │   │   │   ├── entrypoints.py
    │           │   │   │   ├── filesystem.py
    │           │   │   │   ├── filetypes.py
    │           │   │   │   ├── glibc.py
    │           │   │   │   ├── hashes.py
    │           │   │   │   ├── __init__.py
    │           │   │   │   ├── inject_securetransport.py
    │           │   │   │   ├── logging.py
    │           │   │   │   ├── misc.py
    │           │   │   │   ├── models.py
    │           │   │   │   ├── packaging.py
    │           │   │   │   ├── parallel.py
    │           │   │   │   ├── pkg_resources.py
    │           │   │   │   ├── setuptools_build.py
    │           │   │   │   ├── subprocess.py
    │           │   │   │   ├── temp_dir.py
    │           │   │   │   ├── unpacking.py
    │           │   │   │   ├── urls.py
    │           │   │   │   ├── virtualenv.py
    │           │   │   │   └── wheel.py
    │           │   │   ├── vcs
    │           │   │   │   ├── bazaar.py
    │           │   │   │   ├── git.py
    │           │   │   │   ├── __init__.py
    │           │   │   │   ├── mercurial.py
    │           │   │   │   ├── subversion.py
    │           │   │   │   └── versioncontrol.py
    │           │   │   └── wheel_builder.py
    │           │   ├── __main__.py
    │           │   ├── py.typed
    │           │   └── _vendor
    │           │       ├── appdirs.py
    │           │       ├── cachecontrol
    │           │       │   ├── adapter.py
    │           │       │   ├── cache.py
    │           │       │   ├── caches
    │           │       │   │   ├── file_cache.py
    │           │       │   │   ├── __init__.py
    │           │       │   │   └── redis_cache.py
    │           │       │   ├── _cmd.py
    │           │       │   ├── compat.py
    │           │       │   ├── controller.py
    │           │       │   ├── filewrapper.py
    │           │       │   ├── heuristics.py
    │           │       │   ├── __init__.py
    │           │       │   ├── serialize.py
    │           │       │   └── wrapper.py
    │           │       ├── certifi
    │           │       │   ├── cacert.pem
    │           │       │   ├── core.py
    │           │       │   ├── __init__.py
    │           │       │   └── __main__.py
    │           │       ├── chardet
    │           │       │   ├── big5freq.py
    │           │       │   ├── big5prober.py
    │           │       │   ├── chardistribution.py
    │           │       │   ├── charsetgroupprober.py
    │           │       │   ├── charsetprober.py
    │           │       │   ├── cli
    │           │       │   │   ├── chardetect.py
    │           │       │   │   └── __init__.py
    │           │       │   ├── codingstatemachine.py
    │           │       │   ├── compat.py
    │           │       │   ├── cp949prober.py
    │           │       │   ├── enums.py
    │           │       │   ├── escprober.py
    │           │       │   ├── escsm.py
    │           │       │   ├── eucjpprober.py
    │           │       │   ├── euckrfreq.py
    │           │       │   ├── euckrprober.py
    │           │       │   ├── euctwfreq.py
    │           │       │   ├── euctwprober.py
    │           │       │   ├── gb2312freq.py
    │           │       │   ├── gb2312prober.py
    │           │       │   ├── hebrewprober.py
    │           │       │   ├── __init__.py
    │           │       │   ├── jisfreq.py
    │           │       │   ├── jpcntx.py
    │           │       │   ├── langbulgarianmodel.py
    │           │       │   ├── langgreekmodel.py
    │           │       │   ├── langhebrewmodel.py
    │           │       │   ├── langhungarianmodel.py
    │           │       │   ├── langrussianmodel.py
    │           │       │   ├── langthaimodel.py
    │           │       │   ├── langturkishmodel.py
    │           │       │   ├── latin1prober.py
    │           │       │   ├── mbcharsetprober.py
    │           │       │   ├── mbcsgroupprober.py
    │           │       │   ├── mbcssm.py
    │           │       │   ├── metadata
    │           │       │   │   ├── __init__.py
    │           │       │   │   └── languages.py
    │           │       │   ├── sbcharsetprober.py
    │           │       │   ├── sbcsgroupprober.py
    │           │       │   ├── sjisprober.py
    │           │       │   ├── universaldetector.py
    │           │       │   ├── utf8prober.py
    │           │       │   └── version.py
    │           │       ├── colorama
    │           │       │   ├── ansi.py
    │           │       │   ├── ansitowin32.py
    │           │       │   ├── initialise.py
    │           │       │   ├── __init__.py
    │           │       │   ├── win32.py
    │           │       │   └── winterm.py
    │           │       ├── distlib
    │           │       │   ├── _backport
    │           │       │   │   ├── __init__.py
    │           │       │   │   ├── misc.py
    │           │       │   │   ├── shutil.py
    │           │       │   │   ├── sysconfig.cfg
    │           │       │   │   ├── sysconfig.py
    │           │       │   │   └── tarfile.py
    │           │       │   ├── compat.py
    │           │       │   ├── database.py
    │           │       │   ├── index.py
    │           │       │   ├── __init__.py
    │           │       │   ├── locators.py
    │           │       │   ├── manifest.py
    │           │       │   ├── markers.py
    │           │       │   ├── metadata.py
    │           │       │   ├── resources.py
    │           │       │   ├── scripts.py
    │           │       │   ├── t32.exe
    │           │       │   ├── t64.exe
    │           │       │   ├── util.py
    │           │       │   ├── version.py
    │           │       │   ├── w32.exe
    │           │       │   ├── w64.exe
    │           │       │   └── wheel.py
    │           │       ├── distro.py
    │           │       ├── html5lib
    │           │       │   ├── constants.py
    │           │       │   ├── filters
    │           │       │   │   ├── alphabeticalattributes.py
    │           │       │   │   ├── base.py
    │           │       │   │   ├── __init__.py
    │           │       │   │   ├── inject_meta_charset.py
    │           │       │   │   ├── lint.py
    │           │       │   │   ├── optionaltags.py
    │           │       │   │   ├── sanitizer.py
    │           │       │   │   └── whitespace.py
    │           │       │   ├── html5parser.py
    │           │       │   ├── _ihatexml.py
    │           │       │   ├── __init__.py
    │           │       │   ├── _inputstream.py
    │           │       │   ├── serializer.py
    │           │       │   ├── _tokenizer.py
    │           │       │   ├── treeadapters
    │           │       │   │   ├── genshi.py
    │           │       │   │   ├── __init__.py
    │           │       │   │   └── sax.py
    │           │       │   ├── treebuilders
    │           │       │   │   ├── base.py
    │           │       │   │   ├── dom.py
    │           │       │   │   ├── etree_lxml.py
    │           │       │   │   ├── etree.py
    │           │       │   │   └── __init__.py
    │           │       │   ├── treewalkers
    │           │       │   │   ├── base.py
    │           │       │   │   ├── dom.py
    │           │       │   │   ├── etree_lxml.py
    │           │       │   │   ├── etree.py
    │           │       │   │   ├── genshi.py
    │           │       │   │   └── __init__.py
    │           │       │   ├── _trie
    │           │       │   │   ├── _base.py
    │           │       │   │   ├── __init__.py
    │           │       │   │   └── py.py
    │           │       │   └── _utils.py
    │           │       ├── idna
    │           │       │   ├── codec.py
    │           │       │   ├── compat.py
    │           │       │   ├── core.py
    │           │       │   ├── idnadata.py
    │           │       │   ├── __init__.py
    │           │       │   ├── intranges.py
    │           │       │   ├── package_data.py
    │           │       │   └── uts46data.py
    │           │       ├── __init__.py
    │           │       ├── msgpack
    │           │       │   ├── exceptions.py
    │           │       │   ├── ext.py
    │           │       │   ├── fallback.py
    │           │       │   ├── __init__.py
    │           │       │   └── _version.py
    │           │       ├── packaging
    │           │       │   ├── __about__.py
    │           │       │   ├── _compat.py
    │           │       │   ├── __init__.py
    │           │       │   ├── markers.py
    │           │       │   ├── requirements.py
    │           │       │   ├── specifiers.py
    │           │       │   ├── _structures.py
    │           │       │   ├── tags.py
    │           │       │   ├── _typing.py
    │           │       │   ├── utils.py
    │           │       │   └── version.py
    │           │       ├── pep517
    │           │       │   ├── build.py
    │           │       │   ├── check.py
    │           │       │   ├── colorlog.py
    │           │       │   ├── compat.py
    │           │       │   ├── dirtools.py
    │           │       │   ├── envbuild.py
    │           │       │   ├── __init__.py
    │           │       │   ├── in_process
    │           │       │   │   ├── __init__.py
    │           │       │   │   └── _in_process.py
    │           │       │   ├── meta.py
    │           │       │   └── wrappers.py
    │           │       ├── pkg_resources
    │           │       │   ├── __init__.py
    │           │       │   └── py31compat.py
    │           │       ├── progress
    │           │       │   ├── bar.py
    │           │       │   ├── counter.py
    │           │       │   ├── __init__.py
    │           │       │   └── spinner.py
    │           │       ├── pyparsing.py
    │           │       ├── requests
    │           │       │   ├── adapters.py
    │           │       │   ├── api.py
    │           │       │   ├── auth.py
    │           │       │   ├── certs.py
    │           │       │   ├── compat.py
    │           │       │   ├── cookies.py
    │           │       │   ├── exceptions.py
    │           │       │   ├── help.py
    │           │       │   ├── hooks.py
    │           │       │   ├── __init__.py
    │           │       │   ├── _internal_utils.py
    │           │       │   ├── models.py
    │           │       │   ├── packages.py
    │           │       │   ├── sessions.py
    │           │       │   ├── status_codes.py
    │           │       │   ├── structures.py
    │           │       │   ├── utils.py
    │           │       │   └── __version__.py
    │           │       ├── resolvelib
    │           │       │   ├── compat
    │           │       │   │   ├── collections_abc.py
    │           │       │   │   └── __init__.py
    │           │       │   ├── __init__.py
    │           │       │   ├── providers.py
    │           │       │   ├── reporters.py
    │           │       │   ├── resolvers.py
    │           │       │   └── structs.py
    │           │       ├── six.py
    │           │       ├── tenacity
    │           │       │   ├── after.py
    │           │       │   ├── _asyncio.py
    │           │       │   ├── before.py
    │           │       │   ├── before_sleep.py
    │           │       │   ├── compat.py
    │           │       │   ├── __init__.py
    │           │       │   ├── nap.py
    │           │       │   ├── retry.py
    │           │       │   ├── stop.py
    │           │       │   ├── tornadoweb.py
    │           │       │   ├── _utils.py
    │           │       │   └── wait.py
    │           │       ├── toml
    │           │       │   ├── decoder.py
    │           │       │   ├── encoder.py
    │           │       │   ├── __init__.py
    │           │       │   ├── ordered.py
    │           │       │   └── tz.py
    │           │       ├── urllib3
    │           │       │   ├── _collections.py
    │           │       │   ├── connectionpool.py
    │           │       │   ├── connection.py
    │           │       │   ├── contrib
    │           │       │   │   ├── _appengine_environ.py
    │           │       │   │   ├── appengine.py
    │           │       │   │   ├── __init__.py
    │           │       │   │   ├── ntlmpool.py
    │           │       │   │   ├── pyopenssl.py
    │           │       │   │   ├── _securetransport
    │           │       │   │   │   ├── bindings.py
    │           │       │   │   │   ├── __init__.py
    │           │       │   │   │   └── low_level.py
    │           │       │   │   ├── securetransport.py
    │           │       │   │   └── socks.py
    │           │       │   ├── exceptions.py
    │           │       │   ├── fields.py
    │           │       │   ├── filepost.py
    │           │       │   ├── __init__.py
    │           │       │   ├── packages
    │           │       │   │   ├── backports
    │           │       │   │   │   ├── __init__.py
    │           │       │   │   │   └── makefile.py
    │           │       │   │   ├── __init__.py
    │           │       │   │   ├── six.py
    │           │       │   │   └── ssl_match_hostname
    │           │       │   │       ├── _implementation.py
    │           │       │   │       └── __init__.py
    │           │       │   ├── poolmanager.py
    │           │       │   ├── request.py
    │           │       │   ├── response.py
    │           │       │   ├── util
    │           │       │   │   ├── connection.py
    │           │       │   │   ├── __init__.py
    │           │       │   │   ├── proxy.py
    │           │       │   │   ├── queue.py
    │           │       │   │   ├── request.py
    │           │       │   │   ├── response.py
    │           │       │   │   ├── retry.py
    │           │       │   │   ├── ssl_.py
    │           │       │   │   ├── ssltransport.py
    │           │       │   │   ├── timeout.py
    │           │       │   │   ├── url.py
    │           │       │   │   └── wait.py
    │           │       │   └── _version.py
    │           │       ├── vendor.txt
    │           │       └── webencodings
    │           │           ├── __init__.py
    │           │           ├── labels.py
    │           │           ├── mklabels.py
    │           │           ├── tests.py
    │           │           └── x_user_defined.py
    │           ├── pip-21.1.2.dist-info
    │           │   ├── entry_points.txt
    │           │   ├── INSTALLER
    │           │   ├── LICENSE.txt
    │           │   ├── METADATA
    │           │   ├── RECORD
    │           │   ├── top_level.txt
    │           │   └── WHEEL
    │           ├── pip-21.1.2.virtualenv
    │           ├── pkg_resources
    │           │   ├── extern
    │           │   │   ├── __init__.py
    │           │   │   └── __pycache__
    │           │   │       └── __init__.cpython-39.pyc
    │           │   ├── __init__.py
    │           │   ├── __pycache__
    │           │   │   └── __init__.cpython-39.pyc
    │           │   ├── tests
    │           │   │   └── data
    │           │   │       └── my-test-package-source
    │           │   │           └── setup.py
    │           │   └── _vendor
    │           │       ├── appdirs.py
    │           │       ├── __init__.py
    │           │       ├── packaging
    │           │       │   ├── __about__.py
    │           │       │   ├── _compat.py
    │           │       │   ├── __init__.py
    │           │       │   ├── markers.py
    │           │       │   ├── __pycache__
    │           │       │   │   ├── __about__.cpython-39.pyc
    │           │       │   │   ├── _compat.cpython-39.pyc
    │           │       │   │   ├── __init__.cpython-39.pyc
    │           │       │   │   ├── markers.cpython-39.pyc
    │           │       │   │   ├── requirements.cpython-39.pyc
    │           │       │   │   ├── specifiers.cpython-39.pyc
    │           │       │   │   ├── _structures.cpython-39.pyc
    │           │       │   │   ├── _typing.cpython-39.pyc
    │           │       │   │   ├── utils.cpython-39.pyc
    │           │       │   │   └── version.cpython-39.pyc
    │           │       │   ├── requirements.py
    │           │       │   ├── specifiers.py
    │           │       │   ├── _structures.py
    │           │       │   ├── tags.py
    │           │       │   ├── _typing.py
    │           │       │   ├── utils.py
    │           │       │   └── version.py
    │           │       ├── __pycache__
    │           │       │   ├── appdirs.cpython-39.pyc
    │           │       │   ├── __init__.cpython-39.pyc
    │           │       │   └── pyparsing.cpython-39.pyc
    │           │       └── pyparsing.py
    │           ├── __pycache__
    │           │   └── _virtualenv.cpython-39.pyc
    │           ├── setuptools
    │           │   ├── archive_util.py
    │           │   ├── build_meta.py
    │           │   ├── cli-32.exe
    │           │   ├── cli-64.exe
    │           │   ├── cli.exe
    │           │   ├── command
    │           │   │   ├── alias.py
    │           │   │   ├── bdist_egg.py
    │           │   │   ├── bdist_rpm.py
    │           │   │   ├── build_clib.py
    │           │   │   ├── build_ext.py
    │           │   │   ├── build_py.py
    │           │   │   ├── develop.py
    │           │   │   ├── dist_info.py
    │           │   │   ├── easy_install.py
    │           │   │   ├── egg_info.py
    │           │   │   ├── __init__.py
    │           │   │   ├── install_egg_info.py
    │           │   │   ├── install_lib.py
    │           │   │   ├── install.py
    │           │   │   ├── install_scripts.py
    │           │   │   ├── launcher manifest.xml
    │           │   │   ├── py36compat.py
    │           │   │   ├── register.py
    │           │   │   ├── rotate.py
    │           │   │   ├── saveopts.py
    │           │   │   ├── sdist.py
    │           │   │   ├── setopt.py
    │           │   │   ├── test.py
    │           │   │   ├── upload_docs.py
    │           │   │   └── upload.py
    │           │   ├── config.py
    │           │   ├── depends.py
    │           │   ├── _deprecation_warning.py
    │           │   ├── dep_util.py
    │           │   ├── dist.py
    │           │   ├── _distutils
    │           │   │   ├── archive_util.py
    │           │   │   ├── bcppcompiler.py
    │           │   │   ├── ccompiler.py
    │           │   │   ├── cmd.py
    │           │   │   ├── command
    │           │   │   │   ├── bdist_dumb.py
    │           │   │   │   ├── bdist_msi.py
    │           │   │   │   ├── bdist.py
    │           │   │   │   ├── bdist_rpm.py
    │           │   │   │   ├── bdist_wininst.py
    │           │   │   │   ├── build_clib.py
    │           │   │   │   ├── build_ext.py
    │           │   │   │   ├── build.py
    │           │   │   │   ├── build_py.py
    │           │   │   │   ├── build_scripts.py
    │           │   │   │   ├── check.py
    │           │   │   │   ├── clean.py
    │           │   │   │   ├── config.py
    │           │   │   │   ├── __init__.py
    │           │   │   │   ├── install_data.py
    │           │   │   │   ├── install_egg_info.py
    │           │   │   │   ├── install_headers.py
    │           │   │   │   ├── install_lib.py
    │           │   │   │   ├── install.py
    │           │   │   │   ├── install_scripts.py
    │           │   │   │   ├── py37compat.py
    │           │   │   │   ├── register.py
    │           │   │   │   ├── sdist.py
    │           │   │   │   └── upload.py
    │           │   │   ├── config.py
    │           │   │   ├── core.py
    │           │   │   ├── cygwinccompiler.py
    │           │   │   ├── debug.py
    │           │   │   ├── dep_util.py
    │           │   │   ├── dir_util.py
    │           │   │   ├── dist.py
    │           │   │   ├── errors.py
    │           │   │   ├── extension.py
    │           │   │   ├── fancy_getopt.py
    │           │   │   ├── filelist.py
    │           │   │   ├── file_util.py
    │           │   │   ├── __init__.py
    │           │   │   ├── log.py
    │           │   │   ├── msvc9compiler.py
    │           │   │   ├── _msvccompiler.py
    │           │   │   ├── msvccompiler.py
    │           │   │   ├── py35compat.py
    │           │   │   ├── py38compat.py
    │           │   │   ├── spawn.py
    │           │   │   ├── sysconfig.py
    │           │   │   ├── text_file.py
    │           │   │   ├── unixccompiler.py
    │           │   │   ├── util.py
    │           │   │   ├── versionpredicate.py
    │           │   │   └── version.py
    │           │   ├── errors.py
    │           │   ├── extension.py
    │           │   ├── extern
    │           │   │   └── __init__.py
    │           │   ├── glob.py
    │           │   ├── gui-32.exe
    │           │   ├── gui-64.exe
    │           │   ├── gui.exe
    │           │   ├── _imp.py
    │           │   ├── __init__.py
    │           │   ├── installer.py
    │           │   ├── launch.py
    │           │   ├── lib2to3_ex.py
    │           │   ├── monkey.py
    │           │   ├── msvc.py
    │           │   ├── namespaces.py
    │           │   ├── package_index.py
    │           │   ├── py34compat.py
    │           │   ├── sandbox.py
    │           │   ├── script (dev).tmpl
    │           │   ├── script.tmpl
    │           │   ├── ssl_support.py
    │           │   ├── unicode_utils.py
    │           │   ├── _vendor
    │           │   │   ├── __init__.py
    │           │   │   ├── more_itertools
    │           │   │   │   ├── __init__.py
    │           │   │   │   ├── more.py
    │           │   │   │   └── recipes.py
    │           │   │   ├── ordered_set.py
    │           │   │   ├── packaging
    │           │   │   │   ├── __about__.py
    │           │   │   │   ├── _compat.py
    │           │   │   │   ├── __init__.py
    │           │   │   │   ├── markers.py
    │           │   │   │   ├── requirements.py
    │           │   │   │   ├── specifiers.py
    │           │   │   │   ├── _structures.py
    │           │   │   │   ├── tags.py
    │           │   │   │   ├── _typing.py
    │           │   │   │   ├── utils.py
    │           │   │   │   └── version.py
    │           │   │   └── pyparsing.py
    │           │   ├── version.py
    │           │   ├── wheel.py
    │           │   └── windows_support.py
    │           ├── setuptools-57.0.0.dist-info
    │           │   ├── dependency_links.txt
    │           │   ├── entry_points.txt
    │           │   ├── INSTALLER
    │           │   ├── LICENSE
    │           │   ├── METADATA
    │           │   ├── RECORD
    │           │   ├── top_level.txt
    │           │   └── WHEEL
    │           ├── setuptools-57.0.0.virtualenv
    │           ├── _virtualenv.pth
    │           ├── _virtualenv.py
    │           ├── werkzeug
    │           │   ├── datastructures.py
    │           │   ├── datastructures.pyi
    │           │   ├── debug
    │           │   │   ├── console.py
    │           │   │   ├── __init__.py
    │           │   │   ├── __pycache__
    │           │   │   │   ├── console.cpython-39.pyc
    │           │   │   │   ├── __init__.cpython-39.pyc
    │           │   │   │   ├── repr.cpython-39.pyc
    │           │   │   │   └── tbtools.cpython-39.pyc
    │           │   │   ├── repr.py
    │           │   │   ├── shared
    │           │   │   │   ├── console.png
    │           │   │   │   ├── debugger.js
    │           │   │   │   ├── FONT_LICENSE
    │           │   │   │   ├── ICON_LICENSE.md
    │           │   │   │   ├── less.png
    │           │   │   │   ├── more.png
    │           │   │   │   ├── source.png
    │           │   │   │   ├── style.css
    │           │   │   │   └── ubuntu.ttf
    │           │   │   └── tbtools.py
    │           │   ├── exceptions.py
    │           │   ├── filesystem.py
    │           │   ├── formparser.py
    │           │   ├── http.py
    │           │   ├── __init__.py
    │           │   ├── _internal.py
    │           │   ├── local.py
    │           │   ├── middleware
    │           │   │   ├── dispatcher.py
    │           │   │   ├── http_proxy.py
    │           │   │   ├── __init__.py
    │           │   │   ├── lint.py
    │           │   │   ├── profiler.py
    │           │   │   ├── proxy_fix.py
    │           │   │   ├── __pycache__
    │           │   │   │   ├── dispatcher.cpython-39.pyc
    │           │   │   │   ├── http_proxy.cpython-39.pyc
    │           │   │   │   ├── __init__.cpython-39.pyc
    │           │   │   │   ├── lint.cpython-39.pyc
    │           │   │   │   ├── profiler.cpython-39.pyc
    │           │   │   │   ├── proxy_fix.cpython-39.pyc
    │           │   │   │   └── shared_data.cpython-39.pyc
    │           │   │   └── shared_data.py
    │           │   ├── __pycache__
    │           │   │   ├── datastructures.cpython-39.pyc
    │           │   │   ├── exceptions.cpython-39.pyc
    │           │   │   ├── filesystem.cpython-39.pyc
    │           │   │   ├── formparser.cpython-39.pyc
    │           │   │   ├── http.cpython-39.pyc
    │           │   │   ├── __init__.cpython-39.pyc
    │           │   │   ├── _internal.cpython-39.pyc
    │           │   │   ├── local.cpython-39.pyc
    │           │   │   ├── _reloader.cpython-39.pyc
    │           │   │   ├── routing.cpython-39.pyc
    │           │   │   ├── security.cpython-39.pyc
    │           │   │   ├── serving.cpython-39.pyc
    │           │   │   ├── testapp.cpython-39.pyc
    │           │   │   ├── test.cpython-39.pyc
    │           │   │   ├── urls.cpython-39.pyc
    │           │   │   ├── user_agent.cpython-39.pyc
    │           │   │   ├── useragents.cpython-39.pyc
    │           │   │   ├── utils.cpython-39.pyc
    │           │   │   └── wsgi.cpython-39.pyc
    │           │   ├── py.typed
    │           │   ├── _reloader.py
    │           │   ├── routing.py
    │           │   ├── sansio
    │           │   │   ├── __init__.py
    │           │   │   ├── multipart.py
    │           │   │   ├── __pycache__
    │           │   │   │   ├── __init__.cpython-39.pyc
    │           │   │   │   ├── multipart.cpython-39.pyc
    │           │   │   │   ├── request.cpython-39.pyc
    │           │   │   │   ├── response.cpython-39.pyc
    │           │   │   │   └── utils.cpython-39.pyc
    │           │   │   ├── request.py
    │           │   │   ├── response.py
    │           │   │   └── utils.py
    │           │   ├── security.py
    │           │   ├── serving.py
    │           │   ├── testapp.py
    │           │   ├── test.py
    │           │   ├── urls.py
    │           │   ├── user_agent.py
    │           │   ├── useragents.py
    │           │   ├── utils.py
    │           │   ├── wrappers
    │           │   │   ├── accept.py
    │           │   │   ├── auth.py
    │           │   │   ├── base_request.py
    │           │   │   ├── base_response.py
    │           │   │   ├── common_descriptors.py
    │           │   │   ├── cors.py
    │           │   │   ├── etag.py
    │           │   │   ├── __init__.py
    │           │   │   ├── json.py
    │           │   │   ├── __pycache__
    │           │   │   │   ├── accept.cpython-39.pyc
    │           │   │   │   ├── auth.cpython-39.pyc
    │           │   │   │   ├── base_request.cpython-39.pyc
    │           │   │   │   ├── base_response.cpython-39.pyc
    │           │   │   │   ├── common_descriptors.cpython-39.pyc
    │           │   │   │   ├── cors.cpython-39.pyc
    │           │   │   │   ├── etag.cpython-39.pyc
    │           │   │   │   ├── __init__.cpython-39.pyc
    │           │   │   │   ├── json.cpython-39.pyc
    │           │   │   │   ├── request.cpython-39.pyc
    │           │   │   │   ├── response.cpython-39.pyc
    │           │   │   │   └── user_agent.cpython-39.pyc
    │           │   │   ├── request.py
    │           │   │   ├── response.py
    │           │   │   └── user_agent.py
    │           │   └── wsgi.py
    │           ├── Werkzeug-2.0.2.dist-info
    │           │   ├── INSTALLER
    │           │   ├── LICENSE.rst
    │           │   ├── METADATA
    │           │   ├── RECORD
    │           │   ├── top_level.txt
    │           │   └── WHEEL
    │           ├── wheel
    │           │   ├── bdist_wheel.py
    │           │   ├── cli
    │           │   │   ├── convert.py
    │           │   │   ├── __init__.py
    │           │   │   ├── pack.py
    │           │   │   └── unpack.py
    │           │   ├── __init__.py
    │           │   ├── macosx_libfile.py
    │           │   ├── __main__.py
    │           │   ├── metadata.py
    │           │   ├── pkginfo.py
    │           │   ├── util.py
    │           │   ├── vendored
    │           │   │   ├── __init__.py
    │           │   │   └── packaging
    │           │   │       ├── __init__.py
    │           │   │       ├── tags.py
    │           │   │       └── _typing.py
    │           │   └── wheelfile.py
    │           ├── wheel-0.36.2.dist-info
    │           │   ├── entry_points.txt
    │           │   ├── INSTALLER
    │           │   ├── LICENSE.txt
    │           │   ├── METADATA
    │           │   ├── RECORD
    │           │   ├── top_level.txt
    │           │   └── WHEEL
    │           └── wheel-0.36.2.virtualenv
    └── pyvenv.cfg


```


## Flask Configuration

#### Example

```
app = Flask(__name__)
app.config['DEBUG'] = True
```
### Configuring From Files

#### Example Usage

```
app = Flask(__name__ )
app.config.from_pyfile('config.Development.cfg')
```

#### cfg example

```

##Flask settings
DEBUG = True  # True/False
TESTING = False


