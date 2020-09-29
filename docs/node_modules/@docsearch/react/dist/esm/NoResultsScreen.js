function _toConsumableArray(arr) { return _arrayWithoutHoles(arr) || _iterableToArray(arr) || _unsupportedIterableToArray(arr) || _nonIterableSpread(); }

function _nonIterableSpread() { throw new TypeError("Invalid attempt to spread non-iterable instance.\nIn order to be iterable, non-array objects must have a [Symbol.iterator]() method."); }

function _unsupportedIterableToArray(o, minLen) { if (!o) return; if (typeof o === "string") return _arrayLikeToArray(o, minLen); var n = Object.prototype.toString.call(o).slice(8, -1); if (n === "Object" && o.constructor) n = o.constructor.name; if (n === "Map" || n === "Set") return Array.from(o); if (n === "Arguments" || /^(?:Ui|I)nt(?:8|16|32)(?:Clamped)?Array$/.test(n)) return _arrayLikeToArray(o, minLen); }

function _iterableToArray(iter) { if (typeof Symbol !== "undefined" && Symbol.iterator in Object(iter)) return Array.from(iter); }

function _arrayWithoutHoles(arr) { if (Array.isArray(arr)) return _arrayLikeToArray(arr); }

function _arrayLikeToArray(arr, len) { if (len == null || len > arr.length) len = arr.length; for (var i = 0, arr2 = new Array(len); i < len; i++) { arr2[i] = arr[i]; } return arr2; }

import React from 'react';
import { NoResultsIcon } from './icons';
export function NoResultsScreen(props) {
  var searchSuggestions = props.state.context.searchSuggestions;
  return /*#__PURE__*/React.createElement("div", {
    className: "DocSearch-NoResults"
  }, /*#__PURE__*/React.createElement("div", {
    className: "DocSearch-Screen-Icon"
  }, /*#__PURE__*/React.createElement(NoResultsIcon, null)), /*#__PURE__*/React.createElement("p", {
    className: "DocSearch-Title"
  }, "No results for \"", /*#__PURE__*/React.createElement("strong", null, props.state.query), "\""), searchSuggestions && searchSuggestions.length > 0 && /*#__PURE__*/React.createElement("div", {
    className: "DocSearch-NoResults-Prefill-List"
  }, /*#__PURE__*/React.createElement("p", {
    className: "DocSearch-Help"
  }, "Try searching for:"), /*#__PURE__*/React.createElement("ul", null, searchSuggestions.slice(0, 3).reduce(function (acc, search) {
    return [].concat(_toConsumableArray(acc), [/*#__PURE__*/React.createElement("li", {
      key: search
    }, /*#__PURE__*/React.createElement("button", {
      className: "DocSearch-Prefill",
      key: search,
      onClick: function onClick() {
        props.setQuery(search.toLowerCase() + ' ');
        props.refresh();
        props.inputRef.current.focus();
      }
    }, search))]);
  }, []))), /*#__PURE__*/React.createElement("p", {
    className: "DocSearch-Help"
  }, "Believe this query should return results?", ' ', /*#__PURE__*/React.createElement("a", {
    href: "https://github.com/algolia/docsearch-configs/issues/new?template=Missing_results.md&title=[".concat(props.indexName, "]+Missing+results+for+query+\"").concat(props.state.query, "\""),
    target: "_blank",
    rel: "noopener noreferrer"
  }, "Let us know"), "."));
}