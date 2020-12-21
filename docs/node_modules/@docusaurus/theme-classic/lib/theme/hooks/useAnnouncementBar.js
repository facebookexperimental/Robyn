"use strict";

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.default = void 0;

var _react = require("react");

var _themeCommon = require("@docusaurus/theme-common");

/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
const STORAGE_DISMISS_KEY = 'docusaurus.announcement.dismiss';
const STORAGE_ID_KEY = 'docusaurus.announcement.id';

const useAnnouncementBar = () => {
  const {
    announcementBar
  } = (0, _themeCommon.useThemeConfig)();
  const [isClosed, setClosed] = (0, _react.useState)(true);
  const handleClose = (0, _react.useCallback)(() => {
    localStorage.setItem(STORAGE_DISMISS_KEY, 'true');
    setClosed(true);
  }, []);
  (0, _react.useEffect)(() => {
    if (!announcementBar) {
      return;
    }

    const {
      id
    } = announcementBar;
    let viewedId = localStorage.getItem(STORAGE_ID_KEY); // retrocompatibility due to spelling mistake of default id
    // see https://github.com/facebook/docusaurus/issues/3338

    if (viewedId === 'annoucement-bar') {
      viewedId = 'announcement-bar';
    }

    const isNewAnnouncement = id !== viewedId;
    localStorage.setItem(STORAGE_ID_KEY, id);

    if (isNewAnnouncement) {
      localStorage.setItem(STORAGE_DISMISS_KEY, 'false');
    }

    if (isNewAnnouncement || localStorage.getItem(STORAGE_DISMISS_KEY) === 'false') {
      setClosed(false);
    }
  }, []);
  return {
    isAnnouncementBarClosed: isClosed,
    closeAnnouncementBar: handleClose
  };
};

var _default = useAnnouncementBar;
exports.default = _default;