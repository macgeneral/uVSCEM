# uVSCEM
unofficial VSCode Extension Manager

This little program is designed to address some limitations in air-gapped and proxy environments that currently occur due to partially missing proxy support in Visual Studio Code.  
It allows for the installation of extensions in DevContainers, even when a proxy is involved, by downloading and installing them manually via Python's [`requests`](https://requests.readthedocs.io/en/latest/) library.

It's currently a work in progress, and complete offline installation is still missing (integrating it is a minor step), but it should be good enough to be used as `"postAttachCommand"` in `devcontainer.json` for environments where a proxy is involved and where automatic extension installation currently fails due to various errors.

**It's a workaround for the following upstream issues in Visual Studio Code:**
- [ ] [#12588: Extension Proxy Support](https://github.com/microsoft/vscode/issues/12588)
- [ ] [#29910: CLI Proxy Support](https://github.com/microsoft/vscode/issues/29910)
- [ ] [How to make use of "code" cli command in postAttachCommand script](https://github.com/orgs/devcontainers/discussions/94)


## Getting started

In your `Dockerfile` add
```dockerfile
RUN pip install uvscem
```
(preferably after creating a virtual environment and a non-root user)

In your `devcontainer.json`
```
{
  "name": ..
  ..
  "postAttachCommand": "uvscem --config-file /path/to/devcontainer.json"
  ..
}
```

uVSCEM will then install (and update) all extensions listed in `devcontainer.json` each time Visual Studio Code / the DevContainer is started or rebuild.

**Note**
Ensure that your `Dockerfile` contains Python 3.12 and [pip](https://pip.pypa.io/en/stable/), [poetry](https://python-poetry.org), or [pipx](https://pipx.pypa.io/latest/) to install this package. It should theoretically work with Python 3.10 or later but I need to verify this first. Also, confirm that your `PATH` variable includes the (virtual) environment where the package was installed into.

If you have a proxy (e.g. [Cntlm](https://cntlm.sourceforge.net)) listening on localhost it's best to add a line in your `/etc/hosts` or `C:\Windows\system32\drivers\etc\hosts` file:
```
127.0.0.1	localproxy
```
and configure VSCode to use that domain in (User) `settings.json`.
```
{
  ..
  "http.proxy": "http://localproxy:3128"
  ..
}
```
so that some extensions within the DevContainer still got some internet connectivity (If they are proxy aware).



## A big thank you to the following people
- [Jossef Harush Kadouri](http://jossef.com/) for [this GitHub Gist](https://gist.github.com/jossef/8d7681ac0c7fd28e93147aa5044bc129) on how to query the undocumented VisualStudio Code Marketplace API, which I used as blueprint for [`api_client.py`](src/uvscem/api_client.py).
- [Ian McKellar](https://ianloic.com) for his blog post ["VSCode Remote and the command line"](https://ianloic.com/2021/02/16/vscode-remote-and-the-command-line/)  (notable mention: Lazy Ren@Stackoverflow for [this answer](https://stackoverflow.com/a/67916473) pointing me in this direction).
- [Michael Petrov](http://michaelpetrov.com) for [this answer](https://stackoverflow.com/a/62277798) on StackOverflow on how to test if a socket is closed in python.
