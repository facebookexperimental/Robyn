# 3.0.0

* Allow regexp in multiline arrays (#23)
* Add `template` option for regexps, for `eslint --fix` (#23)
* Update eslint to v5.12.0 (#19)

# 2.0.0

* Use the OS's line endings (`\n` on *nix, `\r\n` on Windows) when parsing and fixing comments. This can be configured with the `lineEndings` option. Major version bump as this could be a breaking change for projects.

# 1.2.0

* Add auto fix functionality (eslint `--fix` option) (#12)

# 1.1.0

* Ignore shebangs above header comments to support ESLint 4+ (#11)

# 1.0.0

* Allow RegExp patterns in addition to strings (#2, #4)
* Fix line comment length mismatch issue (#3)

# 0.1.0

* Add config option to read header from file

# 0.0.2

* Initial release
