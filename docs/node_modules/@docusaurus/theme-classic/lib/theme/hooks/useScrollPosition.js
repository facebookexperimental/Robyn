"use strict";

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.default = void 0;

var _react = require("react");

var _ExecutionEnvironment = _interopRequireDefault(require("@docusaurus/ExecutionEnvironment"));

function _interopRequireDefault(obj) { return obj && obj.__esModule ? obj : { default: obj }; }

/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
const getScrollPosition = () => ({
  scrollX: _ExecutionEnvironment.default.canUseDOM ? window.pageXOffset : 0,
  scrollY: _ExecutionEnvironment.default.canUseDOM ? window.pageYOffset : 0
});

const useScrollPosition = (effect, deps = []) => {
  const [scrollPosition, setScrollPosition] = (0, _react.useState)(getScrollPosition());

  const handleScroll = () => {
    const currentScrollPosition = getScrollPosition();
    setScrollPosition(currentScrollPosition);

    if (effect) {
      effect(currentScrollPosition);
    }
  };

  (0, _react.useEffect)(() => {
    const opts = {
      passive: true
    };
    window.addEventListener('scroll', handleScroll, opts);
    return () => window.removeEventListener('scroll', handleScroll, opts);
  }, deps);
  return scrollPosition;
};

var _default = useScrollPosition;
exports.default = _default;