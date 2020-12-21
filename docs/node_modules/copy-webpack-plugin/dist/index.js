"use strict";

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.default = void 0;

var _path = _interopRequireDefault(require("path"));

var _os = _interopRequireDefault(require("os"));

var _crypto = _interopRequireDefault(require("crypto"));

var _webpack = _interopRequireDefault(require("webpack"));

var _schemaUtils = require("schema-utils");

var _pLimit = _interopRequireDefault(require("p-limit"));

var _globby = _interopRequireDefault(require("globby"));

var _findCacheDir = _interopRequireDefault(require("find-cache-dir"));

var _serializeJavascript = _interopRequireDefault(require("serialize-javascript"));

var _cacache = _interopRequireDefault(require("cacache"));

var _loaderUtils = _interopRequireDefault(require("loader-utils"));

var _normalizePath = _interopRequireDefault(require("normalize-path"));

var _globParent = _interopRequireDefault(require("glob-parent"));

var _fastGlob = _interopRequireDefault(require("fast-glob"));

var _package = require("../package.json");

var _options = _interopRequireDefault(require("./options.json"));

var _promisify = require("./utils/promisify");

function _interopRequireDefault(obj) { return obj && obj.__esModule ? obj : { default: obj }; }

// webpack 5 exposes the sources property to ensure the right version of webpack-sources is used
const {
  RawSource
} = // eslint-disable-next-line global-require
_webpack.default.sources || require("webpack-sources");

const template = /(\[ext\])|(\[name\])|(\[path\])|(\[folder\])|(\[emoji(?::(\d+))?\])|(\[(?:([^:\]]+):)?(?:hash|contenthash)(?::([a-z]+\d*))?(?::(\d+))?\])|(\[\d+\])/;

class CopyPlugin {
  constructor(options = {}) {
    (0, _schemaUtils.validate)(_options.default, options, {
      name: "Copy Plugin",
      baseDataPath: "options"
    });
    this.patterns = options.patterns;
    this.options = options.options || {};
  }

  static async createSnapshot(compilation, startTime, dependency) {
    if (!compilation.fileSystemInfo) {
      return;
    } // eslint-disable-next-line consistent-return


    return new Promise((resolve, reject) => {
      compilation.fileSystemInfo.createSnapshot(startTime, [dependency], // eslint-disable-next-line no-undefined
      undefined, // eslint-disable-next-line no-undefined
      undefined, null, (error, snapshot) => {
        if (error) {
          reject(error);
          return;
        }

        resolve(snapshot);
      });
    });
  }

  static async checkSnapshotValid(compilation, snapshot) {
    if (!compilation.fileSystemInfo) {
      return;
    } // eslint-disable-next-line consistent-return


    return new Promise((resolve, reject) => {
      compilation.fileSystemInfo.checkSnapshotValid(snapshot, (error, isValid) => {
        if (error) {
          reject(error);
          return;
        }

        resolve(isValid);
      });
    });
  }

  static async runPattern(compiler, compilation, logger, cache, inputPattern, index) {
    const pattern = typeof inputPattern === "string" ? {
      from: inputPattern
    } : { ...inputPattern
    };
    pattern.fromOrigin = pattern.from;
    pattern.from = _path.default.normalize(pattern.from);
    pattern.compilerContext = compiler.context;
    pattern.context = _path.default.normalize(typeof pattern.context !== "undefined" ? !_path.default.isAbsolute(pattern.context) ? _path.default.join(pattern.compilerContext, pattern.context) : pattern.context : pattern.compilerContext);
    logger.log(`starting to process a pattern from '${pattern.from}' using '${pattern.context}' context`);

    if (_path.default.isAbsolute(pattern.from)) {
      pattern.absoluteFrom = pattern.from;
    } else {
      pattern.absoluteFrom = _path.default.resolve(pattern.context, pattern.from);
    }

    logger.debug(`getting stats for '${pattern.absoluteFrom}'...`);
    const {
      inputFileSystem
    } = compiler;
    let stats;

    try {
      stats = await (0, _promisify.stat)(inputFileSystem, pattern.absoluteFrom);
    } catch (error) {// Nothing
    }

    if (stats) {
      if (stats.isDirectory()) {
        pattern.fromType = "dir";
        logger.debug(`determined '${pattern.absoluteFrom}' is a directory`);
      } else if (stats.isFile()) {
        pattern.fromType = "file";
        logger.debug(`determined '${pattern.absoluteFrom}' is a file`);
      } else {
        logger.debug(`determined '${pattern.absoluteFrom}' is a glob`);
      }
    } // eslint-disable-next-line no-param-reassign


    pattern.globOptions = { ...{
        followSymbolicLinks: true
      },
      ...(pattern.globOptions || {}),
      ...{
        cwd: pattern.context,
        objectMode: true
      }
    }; // TODO remove after drop webpack@4

    if (compiler.webpack && inputFileSystem.lstat && inputFileSystem.stat && inputFileSystem.lstatSync && inputFileSystem.statSync && inputFileSystem.readdir && inputFileSystem.readdirSync) {
      pattern.globOptions.fs = inputFileSystem;
    }

    switch (pattern.fromType) {
      case "dir":
        compilation.contextDependencies.add(pattern.absoluteFrom);
        logger.debug(`added '${pattern.absoluteFrom}' as a context dependency`);
        /* eslint-disable no-param-reassign */

        pattern.context = pattern.absoluteFrom;
        pattern.glob = _path.default.posix.join(_fastGlob.default.escapePath((0, _normalizePath.default)(_path.default.resolve(pattern.absoluteFrom))), "**/*");
        pattern.absoluteFrom = _path.default.join(pattern.absoluteFrom, "**/*");

        if (typeof pattern.globOptions.dot === "undefined") {
          pattern.globOptions.dot = true;
        }
        /* eslint-enable no-param-reassign */


        break;

      case "file":
        compilation.fileDependencies.add(pattern.absoluteFrom);
        logger.debug(`added '${pattern.absoluteFrom}' as a file dependency`);
        /* eslint-disable no-param-reassign */

        pattern.context = _path.default.dirname(pattern.absoluteFrom);
        pattern.glob = _fastGlob.default.escapePath((0, _normalizePath.default)(_path.default.resolve(pattern.absoluteFrom)));

        if (typeof pattern.globOptions.dot === "undefined") {
          pattern.globOptions.dot = true;
        }
        /* eslint-enable no-param-reassign */


        break;

      default:
        {
          const contextDependencies = _path.default.normalize((0, _globParent.default)(pattern.absoluteFrom));

          compilation.contextDependencies.add(contextDependencies);
          logger.debug(`added '${contextDependencies}' as a context dependency`);
          /* eslint-disable no-param-reassign */

          pattern.fromType = "glob";
          pattern.glob = _path.default.isAbsolute(pattern.fromOrigin) ? pattern.fromOrigin : _path.default.posix.join(_fastGlob.default.escapePath((0, _normalizePath.default)(_path.default.resolve(pattern.context))), pattern.fromOrigin);
          /* eslint-enable no-param-reassign */
        }
    }

    logger.log(`begin globbing '${pattern.glob}'...`);
    let paths;

    try {
      paths = await (0, _globby.default)(pattern.glob, pattern.globOptions);
    } catch (error) {
      compilation.errors.push(error);
      return;
    }

    if (paths.length === 0) {
      if (pattern.noErrorOnMissing) {
        logger.log(`finished to process a pattern from '${pattern.from}' using '${pattern.context}' context to '${pattern.to}'`);
        return;
      }

      const missingError = new Error(`unable to locate '${pattern.glob}' glob`);
      compilation.errors.push(missingError);
      return;
    }

    const filteredPaths = (await Promise.all(paths.map(async item => {
      // Exclude directories
      if (!item.dirent.isFile()) {
        return false;
      }

      if (pattern.filter) {
        let isFiltered;

        try {
          isFiltered = await pattern.filter(item.path);
        } catch (error) {
          compilation.errors.push(error);
          return false;
        }

        if (!isFiltered) {
          logger.log(`skip '${item.path}', because it was filtered`);
        }

        return isFiltered ? item : false;
      }

      return item;
    }))).filter(item => item);

    if (filteredPaths.length === 0) {
      // TODO should be error in the next major release
      logger.log(`finished to process a pattern from '${pattern.from}' using '${pattern.context}' context to '${pattern.to}'`);
      return;
    }

    const files = await Promise.all(filteredPaths.map(async item => {
      const from = item.path;
      logger.debug(`found '${from}'`); // `globby`/`fast-glob` return the relative path when the path contains special characters on windows

      const absoluteFilename = _path.default.resolve(pattern.context, from);

      pattern.to = typeof pattern.to !== "function" ? _path.default.normalize(typeof pattern.to !== "undefined" ? pattern.to : "") : await pattern.to({
        context: pattern.context,
        absoluteFilename
      });

      const isToDirectory = _path.default.extname(pattern.to) === "" || pattern.to.slice(-1) === _path.default.sep;

      switch (true) {
        // if toType already exists
        case !!pattern.toType:
          break;

        case template.test(pattern.to):
          pattern.toType = "template";
          break;

        case isToDirectory:
          pattern.toType = "dir";
          break;

        default:
          pattern.toType = "file";
      }

      logger.log(`'to' option '${pattern.to}' determinated as '${pattern.toType}'`);
      const relativeFrom = pattern.flatten ? _path.default.basename(absoluteFilename) : _path.default.relative(pattern.context, absoluteFilename);
      let filename = pattern.toType === "dir" ? _path.default.join(pattern.to, relativeFrom) : pattern.to;

      if (_path.default.isAbsolute(filename)) {
        filename = _path.default.relative(compiler.options.output.path, filename);
      }

      logger.log(`determined that '${from}' should write to '${filename}'`);
      const sourceFilename = (0, _normalizePath.default)(_path.default.relative(pattern.compilerContext, absoluteFilename));
      return {
        absoluteFilename,
        sourceFilename,
        filename
      };
    }));
    let assets;

    try {
      assets = await Promise.all(files.map(async file => {
        const {
          absoluteFilename,
          sourceFilename,
          filename
        } = file;
        const result = {
          absoluteFilename,
          sourceFilename,
          filename,
          force: pattern.force,
          info: typeof pattern.info === "function" ? pattern.info(file) || {} : pattern.info || {}
        }; // If this came from a glob or dir, add it to the file dependencies

        if (pattern.fromType === "dir" || pattern.fromType === "glob") {
          compilation.fileDependencies.add(absoluteFilename);
          logger.debug(`added '${absoluteFilename}' as a file dependency`);
        }

        if (cache) {
          let cacheEntry;
          logger.debug(`getting cache for '${absoluteFilename}'...`);

          try {
            cacheEntry = await cache.getPromise(`${sourceFilename}|${index}`, null);
          } catch (error) {
            compilation.errors.push(error);
            return;
          }

          if (cacheEntry) {
            logger.debug(`found cache for '${absoluteFilename}'...`);
            let isValidSnapshot;
            logger.debug(`checking snapshot on valid for '${absoluteFilename}'...`);

            try {
              isValidSnapshot = await CopyPlugin.checkSnapshotValid(compilation, cacheEntry.snapshot);
            } catch (error) {
              compilation.errors.push(error);
              return;
            }

            if (isValidSnapshot) {
              logger.debug(`snapshot for '${absoluteFilename}' is valid`);
              result.source = cacheEntry.source;
            } else {
              logger.debug(`snapshot for '${absoluteFilename}' is invalid`);
            }
          } else {
            logger.debug(`missed cache for '${absoluteFilename}'`);
          }
        }

        if (!result.source) {
          let startTime;

          if (cache) {
            startTime = Date.now();
          }

          logger.debug(`reading '${absoluteFilename}'...`);
          let data;

          try {
            data = await (0, _promisify.readFile)(inputFileSystem, absoluteFilename);
          } catch (error) {
            compilation.errors.push(error);
            return;
          }

          logger.debug(`read '${absoluteFilename}'`);
          result.source = new RawSource(data);

          if (cache) {
            let snapshot;
            logger.debug(`creating snapshot for '${absoluteFilename}'...`);

            try {
              snapshot = await CopyPlugin.createSnapshot(compilation, startTime, absoluteFilename);
            } catch (error) {
              compilation.errors.push(error);
              return;
            }

            if (snapshot) {
              logger.debug(`created snapshot for '${absoluteFilename}'`);
              logger.debug(`storing cache for '${absoluteFilename}'...`);

              try {
                await cache.storePromise(`${sourceFilename}|${index}`, null, {
                  source: result.source,
                  snapshot
                });
              } catch (error) {
                compilation.errors.push(error);
                return;
              }

              logger.debug(`stored cache for '${absoluteFilename}'`);
            }
          }
        }

        if (pattern.transform) {
          logger.log(`transforming content for '${absoluteFilename}'...`);
          const buffer = result.source.source();

          if (pattern.cacheTransform) {
            const defaultCacheKeys = {
              version: _package.version,
              sourceFilename,
              transform: pattern.transform,
              contentHash: _crypto.default.createHash("md4").update(buffer).digest("hex"),
              index
            };
            const cacheKeys = `transform|${(0, _serializeJavascript.default)(typeof pattern.cacheTransform.keys === "function" ? await pattern.cacheTransform.keys(defaultCacheKeys, absoluteFilename) : { ...defaultCacheKeys,
              ...pattern.cacheTransform.keys
            })}`;
            let cacheItem;
            let cacheDirectory;
            logger.debug(`getting transformation cache for '${absoluteFilename}'...`); // webpack@5 API

            if (cache) {
              cacheItem = cache.getItemCache(cacheKeys, cache.getLazyHashedEtag(result.source));
              result.source = await cacheItem.getPromise();
            } else {
              cacheDirectory = pattern.cacheTransform.directory ? pattern.cacheTransform.directory : typeof pattern.cacheTransform === "string" ? pattern.cacheTransform : (0, _findCacheDir.default)({
                name: "copy-webpack-plugin"
              }) || _os.default.tmpdir();
              let cached;

              try {
                cached = await _cacache.default.get(cacheDirectory, cacheKeys);
              } catch (error) {
                logger.debug(`no transformation cache for '${absoluteFilename}'...`);
              } // eslint-disable-next-line no-undefined


              result.source = cached ? new RawSource(cached.data) : undefined;
            }

            logger.debug(result.source ? `found transformation cache for '${absoluteFilename}'` : `no transformation cache for '${absoluteFilename}'`);

            if (!result.source) {
              const transformed = await pattern.transform(buffer, absoluteFilename);
              result.source = new RawSource(transformed);
              logger.debug(`caching transformation for '${absoluteFilename}'...`); // webpack@5 API

              if (cache) {
                await cacheItem.storePromise(result.source);
              } else {
                try {
                  await _cacache.default.put(cacheDirectory, cacheKeys, transformed);
                } catch (error) {
                  compilation.errors.push(error);
                  return;
                }
              }

              logger.debug(`cached transformation for '${absoluteFilename}'`);
            }
          } else {
            result.source = new RawSource(await pattern.transform(buffer, absoluteFilename));
          }
        }

        if (pattern.toType === "template") {
          logger.log(`interpolating template '${filename}' for '${sourceFilename}'...`); // If it doesn't have an extension, remove it from the pattern
          // ie. [name].[ext] or [name][ext] both become [name]

          if (!_path.default.extname(absoluteFilename)) {
            // eslint-disable-next-line no-param-reassign
            result.filename = result.filename.replace(/\.?\[ext]/g, "");
          } // eslint-disable-next-line no-param-reassign


          result.immutable = /\[(?:([^:\]]+):)?(?:hash|contenthash)(?::([a-z]+\d*))?(?::(\d+))?\]/gi.test(result.filename); // eslint-disable-next-line no-param-reassign

          result.filename = _loaderUtils.default.interpolateName({
            resourcePath: absoluteFilename
          }, result.filename, {
            content: result.source.source(),
            context: pattern.context
          }); // Bug in `loader-utils`, package convert `\\` to `/`, need fix in loader-utils
          // eslint-disable-next-line no-param-reassign

          result.filename = _path.default.normalize(result.filename);
          logger.log(`interpolated template '${filename}' for '${sourceFilename}'`);
        }

        if (pattern.transformPath) {
          logger.log(`transforming '${result.filename}' for '${absoluteFilename}'...`); // eslint-disable-next-line no-param-reassign

          result.immutable = false; // eslint-disable-next-line no-param-reassign

          result.filename = await pattern.transformPath(result.filename, absoluteFilename);
          logger.log(`transformed new '${result.filename}' for '${absoluteFilename}'...`);
        } // eslint-disable-next-line no-param-reassign


        result.filename = (0, _normalizePath.default)(result.filename); // eslint-disable-next-line consistent-return

        return result;
      }));
    } catch (error) {
      compilation.errors.push(error);
      return;
    }

    logger.log(`finished to process a pattern from '${pattern.from}' using '${pattern.context}' context to '${pattern.to}'`); // eslint-disable-next-line consistent-return

    return assets;
  }

  apply(compiler) {
    const pluginName = this.constructor.name;
    const limit = (0, _pLimit.default)(this.options.concurrency || 100);
    compiler.hooks.thisCompilation.tap(pluginName, compilation => {
      const logger = compilation.getLogger("copy-webpack-plugin");
      const cache = compilation.getCache ? compilation.getCache("CopyWebpackPlugin") : // eslint-disable-next-line no-undefined
      undefined;
      compilation.hooks.additionalAssets.tapAsync("copy-webpack-plugin", async callback => {
        logger.log("starting to add additional assets...");
        let assets;

        try {
          assets = await Promise.all(this.patterns.map((item, index) => limit(async () => CopyPlugin.runPattern(compiler, compilation, logger, cache, item, index))));
        } catch (error) {
          compilation.errors.push(error);
          callback();
          return;
        } // Avoid writing assets inside `p-limit`, because it creates concurrency.
        // It could potentially lead to an error - 'Multiple assets emit different content to the same filename'


        assets.reduce((acc, val) => acc.concat(val), []).filter(Boolean).forEach(asset => {
          const {
            absoluteFilename,
            sourceFilename,
            filename,
            source,
            force
          } = asset; // For old version webpack 4

          /* istanbul ignore if */

          if (typeof compilation.emitAsset !== "function") {
            // eslint-disable-next-line no-param-reassign
            compilation.assets[filename] = source;
            return;
          }

          const existingAsset = compilation.getAsset(filename);

          if (existingAsset) {
            if (force) {
              const info = {
                copied: true,
                sourceFilename
              };

              if (asset.immutable) {
                info.immutable = true;
              }

              logger.log(`force updating '${filename}' from '${absoluteFilename}' to compilation assets, because it already exists...`);
              compilation.updateAsset(filename, source, { ...info,
                ...asset.info
              });
              logger.log(`force updated '${filename}' from '${absoluteFilename}' to compilation assets, because it already exists`);
              return;
            }

            logger.log(`skip adding '${filename}' from '${absoluteFilename}' to compilation assets, because it already exists`);
            return;
          }

          const info = {
            copied: true,
            sourceFilename
          };

          if (asset.immutable) {
            info.immutable = true;
          }

          logger.log(`writing '${filename}' from '${absoluteFilename}' to compilation assets...`);
          compilation.emitAsset(filename, source, { ...info,
            ...asset.info
          });
          logger.log(`written '${filename}' from '${absoluteFilename}' to compilation assets`);
        });
        logger.log("finished to adding additional assets");
        callback();
      });

      if (compilation.hooks.statsPrinter) {
        compilation.hooks.statsPrinter.tap(pluginName, stats => {
          stats.hooks.print.for("asset.info.copied").tap("copy-webpack-plugin", (copied, {
            green,
            formatFlag
          }) => // eslint-disable-next-line no-undefined
          copied ? green(formatFlag("copied")) : undefined);
        });
      }
    });
  }

}

var _default = CopyPlugin;
exports.default = _default;