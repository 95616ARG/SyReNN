# https://stackoverflow.com/questions/2537271
genrule(
    name = "build_openssl",
    srcs = glob(["**"]),
    outs = ["openssl-installdir"],
    cmd = """
    # First, we need to copy the OpenSSL source to a writeable location
    # (following links). If we try to do it directly in the original
    # OPENSSLDIR, it consistently throws errors about file system permissions
    # (I'm not entirely sure why, but this fixes it).
    HOMEDIR=$$PWD
    SOURCEDIR=$$PWD/$$(dirname $(location config))
    cp -Lr $$SOURCEDIR openssl
    SOURCEDIR=$$PWD/openssl

    # Then create the install dir.
    INSTALLDIR=$$PWD/openssl-installdir
    mkdir -p $$INSTALLDIR

    # Install OpenSSL to installdir.
    cd $$SOURCEDIR
    ./config shared -fPIC --prefix=$$INSTALLDIR --openssldir=$$INSTALLDIR/openssl > /dev/null
    make -j2 > /dev/null
    make install > /dev/null
    cd $$HOMEDIR

    # And copy to the output directory.
    cp -r $$INSTALLDIR $@
    """,
    visibility = ["//visibility:public"],
)
