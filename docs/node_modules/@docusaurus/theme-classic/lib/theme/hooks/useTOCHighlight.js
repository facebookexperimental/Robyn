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
function useTOCHighlight(linkClassName, linkActiveClassName, topOffset) {
  const [lastActiveLink, setLastActiveLink] = (0, _react.useState)(undefined);
  (0, _react.useEffect)(() => {
    function setActiveLink() {
      function getActiveHeaderAnchor() {
        const headersAnchors = Array.from(document.getElementsByClassName('anchor'));
        const firstAnchorUnderViewportTop = headersAnchors.find(anchor => {
          const {
            top
          } = anchor.getBoundingClientRect();
          return top >= topOffset;
        });

        if (firstAnchorUnderViewportTop) {
          // If first anchor in viewport is under a certain threshold, we consider it's not the active anchor.
          // In such case, the active anchor is the previous one (if it exists), that may be above the viewport
          if (firstAnchorUnderViewportTop.getBoundingClientRect().top >= topOffset) {
            const previousAnchor = headersAnchors[headersAnchors.indexOf(firstAnchorUnderViewportTop) - 1];
            return previousAnchor !== null && previousAnchor !== void 0 ? previousAnchor : firstAnchorUnderViewportTop;
          } // If the anchor is at the top of the viewport, we consider it's the first anchor
          else {
              return firstAnchorUnderViewportTop;
            }
        } // no anchor under viewport top? (ie we are at the bottom of the page)
        else {
            // highlight the last anchor found
            return headersAnchors[headersAnchors.length - 1];
          }
      }

      const activeHeaderAnchor = getActiveHeaderAnchor();

      if (activeHeaderAnchor) {
        let index = 0;
        let itemHighlighted = false; // @ts-expect-error: Must be <a> tags.

        const links = document.getElementsByClassName(linkClassName);

        while (index < links.length && !itemHighlighted) {
          const link = links[index];
          const {
            href
          } = link;
          const anchorValue = decodeURIComponent(href.substring(href.indexOf('#') + 1));

          if (activeHeaderAnchor.id === anchorValue) {
            if (lastActiveLink) {
              lastActiveLink.classList.remove(linkActiveClassName);
            }

            link.classList.add(linkActiveClassName);
            setLastActiveLink(link);
            itemHighlighted = true;
          }

          index += 1;
        }
      }
    }

    document.addEventListener('scroll', setActiveLink);
    document.addEventListener('resize', setActiveLink);
    setActiveLink();
    return () => {
      document.removeEventListener('scroll', setActiveLink);
      document.removeEventListener('resize', setActiveLink);
    };
  });
}

var _default = useTOCHighlight;
exports.default = _default;