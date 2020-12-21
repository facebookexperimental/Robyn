"use strict";

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.default = void 0;

var _react = require("react");

/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
const TAB_CHOICE_PREFIX = 'docusaurus.tab.';

const useTabGroupChoice = () => {
  const [tabGroupChoices, setChoices] = (0, _react.useState)({});
  const setChoiceSyncWithLocalStorage = (0, _react.useCallback)((groupId, newChoice) => {
    try {
      localStorage.setItem(`${TAB_CHOICE_PREFIX}${groupId}`, newChoice);
    } catch (err) {
      console.error(err);
    }
  }, []);
  (0, _react.useEffect)(() => {
    try {
      const localStorageChoices = {};

      for (let i = 0; i < localStorage.length; i += 1) {
        const storageKey = localStorage.key(i);

        if (storageKey.startsWith(TAB_CHOICE_PREFIX)) {
          const groupId = storageKey.substring(TAB_CHOICE_PREFIX.length);
          localStorageChoices[groupId] = localStorage.getItem(storageKey);
        }
      }

      setChoices(localStorageChoices);
    } catch (err) {
      console.error(err);
    }
  }, []);
  return {
    tabGroupChoices,
    setTabGroupChoices: (groupId, newChoice) => {
      setChoices(oldChoices => ({ ...oldChoices,
        [groupId]: newChoice
      }));
      setChoiceSyncWithLocalStorage(groupId, newChoice);
    }
  };
};

var _default = useTabGroupChoice;
exports.default = _default;