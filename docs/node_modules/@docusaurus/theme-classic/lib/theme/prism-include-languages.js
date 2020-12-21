"use strict";

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.default = void 0;

var _ExecutionEnvironment = _interopRequireDefault(require("@docusaurus/ExecutionEnvironment"));

var _docusaurus = _interopRequireDefault(require("@generated/docusaurus.config"));

function _interopRequireDefault(obj) { return obj && obj.__esModule ? obj : { default: obj }; }

/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
const prismIncludeLanguages = PrismObject => {
  if (_ExecutionEnvironment.default.canUseDOM) {
    const {
      themeConfig: {
        prism: {
          additionalLanguages = []
        } = {}
      }
    } = _docusaurus.default;
    window.Prism = PrismObject;
    additionalLanguages.forEach(lang => {
      require(`prismjs/components/prism-${lang}`); // eslint-disable-line

    });
    delete window.Prism;
  }
};

var _default = prismIncludeLanguages;
exports.default = _default;