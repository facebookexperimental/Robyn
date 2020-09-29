import React from 'react';
import { ErrorIcon } from './icons';
export function ErrorScreen() {
  return /*#__PURE__*/React.createElement("div", {
    className: "DocSearch-ErrorScreen"
  }, /*#__PURE__*/React.createElement("div", {
    className: "DocSearch-Screen-Icon"
  }, /*#__PURE__*/React.createElement(ErrorIcon, null)), /*#__PURE__*/React.createElement("p", {
    className: "DocSearch-Title"
  }, "Unable to fetch results"), /*#__PURE__*/React.createElement("p", {
    className: "DocSearch-Help"
  }, "You might want to check your network connection."));
}