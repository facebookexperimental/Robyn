"use strict";

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.default = void 0;

var _react = require("react");

var _ExecutionEnvironment = _interopRequireDefault(require("@docusaurus/ExecutionEnvironment"));

var _themeCommon = require("@docusaurus/theme-common");

function _interopRequireDefault(obj) { return obj && obj.__esModule ? obj : { default: obj }; }

/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
const themes = {
  light: 'light',
  dark: 'dark'
}; // Ensure to always return a valid theme even if input is invalid

const coerceToTheme = theme => {
  return theme === themes.dark ? themes.dark : themes.light;
};

const getInitialTheme = () => {
  if (!_ExecutionEnvironment.default.canUseDOM) {
    return themes.light; // SSR: we don't care
  }

  return coerceToTheme(document.documentElement.getAttribute('data-theme'));
};

const storeTheme = newTheme => {
  try {
    localStorage.setItem('theme', coerceToTheme(newTheme));
  } catch (err) {
    console.error(err);
  }
};

const useTheme = () => {
  const {
    colorMode: {
      disableSwitch,
      respectPrefersColorScheme
    }
  } = (0, _themeCommon.useThemeConfig)();
  const [theme, setTheme] = (0, _react.useState)(getInitialTheme);
  const setLightTheme = (0, _react.useCallback)(() => {
    setTheme(themes.light);
    storeTheme(themes.light);
  }, []);
  const setDarkTheme = (0, _react.useCallback)(() => {
    setTheme(themes.dark);
    storeTheme(themes.dark);
  }, []);
  (0, _react.useEffect)(() => {
    document.documentElement.setAttribute('data-theme', coerceToTheme(theme));
  }, [theme]);
  (0, _react.useEffect)(() => {
    if (disableSwitch) {
      return;
    }

    try {
      const localStorageTheme = localStorage.getItem('theme');

      if (localStorageTheme !== null) {
        setTheme(coerceToTheme(localStorageTheme));
      }
    } catch (err) {
      console.error(err);
    }
  }, [setTheme]);
  (0, _react.useEffect)(() => {
    if (disableSwitch && !respectPrefersColorScheme) {
      return;
    }

    window.matchMedia('(prefers-color-scheme: dark)').addListener(({
      matches
    }) => {
      setTheme(matches ? themes.dark : themes.light);
    });
  }, []);
  return {
    isDarkTheme: theme === themes.dark,
    setLightTheme,
    setDarkTheme
  };
};

var _default = useTheme;
exports.default = _default;