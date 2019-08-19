# https://superuser.com/questions/1346141
# There's another one I lost about finding the current script's directory.
genrule(
    name = "build_python",
    srcs = glob(["**"]) + ["@openssl//:openssl-installdir"],
    outs = ["installdir"],
    cmd = """
    HOMEDIR=$$PWD
    INSTALLDIR=$$PWD/installdir
    cp -Lr $(location @openssl//:openssl-installdir) $$INSTALLDIR

    # See external/openssl.BUILD for notes on why we do this weird copy.
    SOURCEDIR=$$PWD/$$(dirname $(location configure))
    cp -Lr $$SOURCEDIR $$PWD/python
    SOURCEDIR=$$PWD/python

    export LDFLAGS="-L$$INSTALLDIR/lib/ -L$$INSTALLDIR/lib64/"
    export LD_LIBRARY_PATH="$$INSTALLDIR/lib/:$$INSTALLDIR/lib64/"
    export CPPFLAGS="-I$$INSTALLDIR/include -I$$INSTALLDIR/include/openssl"

    # Install Python to installdir.
    cd $$SOURCEDIR
    ./configure --prefix=$$INSTALLDIR --with-openssl=$$INSTALLDIR > /dev/null
    make -j2 > /dev/null
    make install > /dev/null
    cd $$HOMEDIR
    cp -r installdir $(@D)
    """,
    visibility = ["//visibility:public"],
)
