"use strict";

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.resolveTargets = resolveTargets;

function _path() {
  const data = _interopRequireDefault(require("path"));

  _path = function () {
    return data;
  };

  return data;
}

function _helperCompilationTargets() {
  const data = _interopRequireDefault(require("@babel/helper-compilation-targets"));

  _helperCompilationTargets = function () {
    return data;
  };

  return data;
}

function _interopRequireDefault(obj) { return obj && obj.__esModule ? obj : { default: obj }; }

({});

function resolveTargets(options, root, filename) {
  let {
    targets
  } = options;

  if (typeof targets === "string" || Array.isArray(targets)) {
    targets = {
      browsers: targets
    };
  }

  if (targets && targets.esmodules) {
    targets = Object.assign({}, targets, {
      esmodules: "intersect"
    });
  }

  let configFile;

  if (typeof options.browserslistConfigFile === "string") {
    configFile = _path().default.resolve(root, options.browserslistConfigFile);
  }

  return (0, _helperCompilationTargets().default)(targets, {
    ignoreBrowserslistConfig: options.browserslistConfigFile === false,
    configFile,
    configPath: filename != null ? filename : root,
    browserslistEnv: options.browserslistEnv
  });
}