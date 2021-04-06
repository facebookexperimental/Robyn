/*! @algolia/autocomplete-preset-algolia 1.0.0-alpha.44 | MIT License | © Algolia, Inc. and contributors | https://github.com/algolia/autocomplete */
(function (global, factory) {
  typeof exports === 'object' && typeof module !== 'undefined' ? factory(exports) :
  typeof define === 'function' && define.amd ? define(['exports'], factory) :
  (global = typeof globalThis !== 'undefined' ? globalThis : global || self, factory(global['@algolia/autocomplete-preset-algolia'] = {}));
}(this, (function (exports) { 'use strict';

  function _defineProperty(obj, key, value) {
    if (key in obj) {
      Object.defineProperty(obj, key, {
        value: value,
        enumerable: true,
        configurable: true,
        writable: true
      });
    } else {
      obj[key] = value;
    }

    return obj;
  }

  function ownKeys(object, enumerableOnly) {
    var keys = Object.keys(object);

    if (Object.getOwnPropertySymbols) {
      var symbols = Object.getOwnPropertySymbols(object);
      if (enumerableOnly) symbols = symbols.filter(function (sym) {
        return Object.getOwnPropertyDescriptor(object, sym).enumerable;
      });
      keys.push.apply(keys, symbols);
    }

    return keys;
  }

  function _objectSpread2(target) {
    for (var i = 1; i < arguments.length; i++) {
      var source = arguments[i] != null ? arguments[i] : {};

      if (i % 2) {
        ownKeys(Object(source), true).forEach(function (key) {
          _defineProperty(target, key, source[key]);
        });
      } else if (Object.getOwnPropertyDescriptors) {
        Object.defineProperties(target, Object.getOwnPropertyDescriptors(source));
      } else {
        ownKeys(Object(source)).forEach(function (key) {
          Object.defineProperty(target, key, Object.getOwnPropertyDescriptor(source, key));
        });
      }
    }

    return target;
  }

  function _toConsumableArray(arr) {
    return _arrayWithoutHoles(arr) || _iterableToArray(arr) || _unsupportedIterableToArray(arr) || _nonIterableSpread();
  }

  function _arrayWithoutHoles(arr) {
    if (Array.isArray(arr)) return _arrayLikeToArray(arr);
  }

  function _iterableToArray(iter) {
    if (typeof Symbol !== "undefined" && Symbol.iterator in Object(iter)) return Array.from(iter);
  }

  function _unsupportedIterableToArray(o, minLen) {
    if (!o) return;
    if (typeof o === "string") return _arrayLikeToArray(o, minLen);
    var n = Object.prototype.toString.call(o).slice(8, -1);
    if (n === "Object" && o.constructor) n = o.constructor.name;
    if (n === "Map" || n === "Set") return Array.from(o);
    if (n === "Arguments" || /^(?:Ui|I)nt(?:8|16|32)(?:Clamped)?Array$/.test(n)) return _arrayLikeToArray(o, minLen);
  }

  function _arrayLikeToArray(arr, len) {
    if (len == null || len > arr.length) len = arr.length;

    for (var i = 0, arr2 = new Array(len); i < len; i++) arr2[i] = arr[i];

    return arr2;
  }

  function _nonIterableSpread() {
    throw new TypeError("Invalid attempt to spread non-iterable instance.\nIn order to be iterable, non-array objects must have a [Symbol.iterator]() method.");
  }

  function getAttributeValueByPath(record, path) {
    return path.reduce(function (current, key) {
      return current && current[key];
    }, record);
  }

  var warnCache = {
    current: {}
  };
  /**
   * Logs a warning if the condition is not met.
   * This is used to log issues in development environment only.
   */

  function warn(condition, message) {

    if (condition) {
      return;
    }

    var sanitizedMessage = message.trim();
    var hasAlreadyPrinted = warnCache.current[sanitizedMessage];

    if (!hasAlreadyPrinted) {
      warnCache.current[sanitizedMessage] = true; // eslint-disable-next-line no-console

      console.warn("[Autocomplete] ".concat(sanitizedMessage));
    }
  }

  var HIGHLIGHT_PRE_TAG = '__aa-highlight__';
  var HIGHLIGHT_POST_TAG = '__/aa-highlight__';

  /**
   * Creates a data structure that allows to concatenate similar highlighting
   * parts in a single value.
   */
  function createAttributeSet() {
    var initialValue = arguments.length > 0 && arguments[0] !== undefined ? arguments[0] : [];
    var value = initialValue;
    return {
      get: function get() {
        return value;
      },
      add: function add(part) {
        var lastPart = value[value.length - 1];

        if ((lastPart === null || lastPart === void 0 ? void 0 : lastPart.isHighlighted) === part.isHighlighted) {
          value[value.length - 1] = {
            value: lastPart.value + part.value,
            isHighlighted: lastPart.isHighlighted
          };
        } else {
          value.push(part);
        }
      }
    };
  }

  function parseAttribute(_ref) {
    var highlightedValue = _ref.highlightedValue;
    var preTagParts = highlightedValue.split(HIGHLIGHT_PRE_TAG);
    var firstValue = preTagParts.shift();
    var parts = createAttributeSet(firstValue ? [{
      value: firstValue,
      isHighlighted: false
    }] : []);
    preTagParts.forEach(function (part) {
      var postTagParts = part.split(HIGHLIGHT_POST_TAG);
      parts.add({
        value: postTagParts[0],
        isHighlighted: true
      });

      if (postTagParts[1] !== '') {
        parts.add({
          value: postTagParts[1],
          isHighlighted: false
        });
      }
    });
    return parts.get();
  }

  function parseAlgoliaHitHighlight(_ref) {
    var hit = _ref.hit,
        attribute = _ref.attribute;
    var path = Array.isArray(attribute) ? attribute : [attribute];
    var highlightedValue = getAttributeValueByPath(hit, ['_highlightResult'].concat(_toConsumableArray(path), ['value']));

    if (typeof highlightedValue !== 'string') {
      "development" !== 'production' ? warn(false, "The attribute \"".concat(path.join('.'), "\" described by the path ").concat(JSON.stringify(path), " does not exist on the hit. Did you set it in `attributesToHighlight`?") + '\nSee https://www.algolia.com/doc/api-reference/api-parameters/attributesToHighlight/') : void 0;
      highlightedValue = getAttributeValueByPath(hit, path) || '';
    }

    return parseAttribute({
      highlightedValue: highlightedValue
    });
  }

  var htmlEscapes = {
    '&amp;': '&',
    '&lt;': '<',
    '&gt;': '>',
    '&quot;': '"',
    '&#39;': "'"
  };
  var hasAlphanumeric = new RegExp(/\w/i);
  var regexEscapedHtml = /&(amp|quot|lt|gt|#39);/g;
  var regexHasEscapedHtml = RegExp(regexEscapedHtml.source);

  function unescape(value) {
    return value && regexHasEscapedHtml.test(value) ? value.replace(regexEscapedHtml, function (character) {
      return htmlEscapes[character];
    }) : value;
  }

  function isPartHighlighted(parts, i) {
    var _parts, _parts2;

    var current = parts[i];
    var isNextHighlighted = ((_parts = parts[i + 1]) === null || _parts === void 0 ? void 0 : _parts.isHighlighted) || true;
    var isPreviousHighlighted = ((_parts2 = parts[i - 1]) === null || _parts2 === void 0 ? void 0 : _parts2.isHighlighted) || true;

    if (!hasAlphanumeric.test(unescape(current.value)) && isPreviousHighlighted === isNextHighlighted) {
      return isPreviousHighlighted;
    }

    return current.isHighlighted;
  }

  function reverseHighlightedParts(parts) {
    // We don't want to highlight the whole word when no parts match.
    if (!parts.some(function (part) {
      return part.isHighlighted;
    })) {
      return parts.map(function (part) {
        return _objectSpread2(_objectSpread2({}, part), {}, {
          isHighlighted: false
        });
      });
    }

    return parts.map(function (part, i) {
      return _objectSpread2(_objectSpread2({}, part), {}, {
        isHighlighted: !isPartHighlighted(parts, i)
      });
    });
  }

  function parseAlgoliaHitReverseHighlight(props) {
    return reverseHighlightedParts(parseAlgoliaHitHighlight(props));
  }

  function parseAlgoliaHitSnippet(_ref) {
    var hit = _ref.hit,
        attribute = _ref.attribute;
    var path = Array.isArray(attribute) ? attribute : [attribute];
    var highlightedValue = getAttributeValueByPath(hit, ['_snippetResult'].concat(_toConsumableArray(path), ['value']));

    if (typeof highlightedValue !== 'string') {
      "development" !== 'production' ? warn(false, "The attribute \"".concat(path.join('.'), "\" described by the path ").concat(JSON.stringify(path), " does not exist on the hit. Did you set it in `attributesToSnippet`?") + '\nSee https://www.algolia.com/doc/api-reference/api-parameters/attributesToSnippet/') : void 0;
      highlightedValue = getAttributeValueByPath(hit, path) || '';
    }

    return parseAttribute({
      highlightedValue: highlightedValue
    });
  }

  function parseAlgoliaHitReverseSnippet(props) {
    return reverseHighlightedParts(parseAlgoliaHitSnippet(props));
  }

  var version = '1.0.0-alpha.44';

  function searchForFacetValues(_ref) {
    var searchClient = _ref.searchClient,
        queries = _ref.queries,
        _ref$userAgents = _ref.userAgents,
        userAgents = _ref$userAgents === void 0 ? [] : _ref$userAgents;

    if (typeof searchClient.addAlgoliaAgent === 'function') {
      var algoliaAgents = [{
        segment: 'autocomplete-core',
        version: version
      }].concat(_toConsumableArray(userAgents));
      algoliaAgents.forEach(function (_ref2) {
        var segment = _ref2.segment,
            version = _ref2.version;
        searchClient.addAlgoliaAgent(segment, version);
      });
    }

    return searchClient.searchForFacetValues(queries.map(function (searchParameters) {
      var indexName = searchParameters.indexName,
          params = searchParameters.params;
      return {
        indexName: indexName,
        params: _objectSpread2({
          highlightPreTag: HIGHLIGHT_PRE_TAG,
          highlightPostTag: HIGHLIGHT_POST_TAG
        }, params)
      };
    }));
  }

  function getAlgoliaFacetHits(_ref) {
    var searchClient = _ref.searchClient,
        queries = _ref.queries,
        userAgents = _ref.userAgents;
    return searchForFacetValues({
      searchClient: searchClient,
      queries: queries,
      userAgents: userAgents
    }).then(function (response) {
      return response.map(function (result) {
        return result.facetHits.map(function (facetHit) {
          return {
            label: facetHit.value,
            count: facetHit.count,
            _highlightResult: {
              label: {
                value: facetHit.highlighted
              }
            }
          };
        });
      });
    });
  }

  function search(_ref) {
    var searchClient = _ref.searchClient,
        queries = _ref.queries,
        _ref$userAgents = _ref.userAgents,
        userAgents = _ref$userAgents === void 0 ? [] : _ref$userAgents;

    if (typeof searchClient.addAlgoliaAgent === 'function') {
      var algoliaAgents = [{
        segment: 'autocomplete-core',
        version: version
      }].concat(_toConsumableArray(userAgents));
      algoliaAgents.forEach(function (_ref2) {
        var segment = _ref2.segment,
            version = _ref2.version;
        searchClient.addAlgoliaAgent(segment, version);
      });
    }

    return searchClient.search(queries.map(function (searchParameters) {
      var indexName = searchParameters.indexName,
          query = searchParameters.query,
          params = searchParameters.params;
      return {
        indexName: indexName,
        query: query,
        params: _objectSpread2({
          hitsPerPage: 5,
          highlightPreTag: HIGHLIGHT_PRE_TAG,
          highlightPostTag: HIGHLIGHT_POST_TAG
        }, params)
      };
    }));
  }

  function getAlgoliaHits(_ref) {
    var searchClient = _ref.searchClient,
        queries = _ref.queries,
        userAgents = _ref.userAgents;
    return search({
      searchClient: searchClient,
      queries: queries,
      userAgents: userAgents
    }).then(function (response) {
      var results = response.results;
      return results.map(function (result) {
        return result.hits.map(function (hit) {
          return _objectSpread2(_objectSpread2({}, hit), {}, {
            __autocomplete_indexName: result.index,
            __autocomplete_queryID: result.queryID
          });
        });
      });
    });
  }

  function getAlgoliaResults(_ref) {
    var searchClient = _ref.searchClient,
        queries = _ref.queries,
        userAgents = _ref.userAgents;
    return search({
      searchClient: searchClient,
      queries: queries,
      userAgents: userAgents
    }).then(function (response) {
      return response.results;
    });
  }

  exports.getAlgoliaFacetHits = getAlgoliaFacetHits;
  exports.getAlgoliaHits = getAlgoliaHits;
  exports.getAlgoliaResults = getAlgoliaResults;
  exports.parseAlgoliaHitHighlight = parseAlgoliaHitHighlight;
  exports.parseAlgoliaHitReverseHighlight = parseAlgoliaHitReverseHighlight;
  exports.parseAlgoliaHitReverseSnippet = parseAlgoliaHitReverseSnippet;
  exports.parseAlgoliaHitSnippet = parseAlgoliaHitSnippet;

  Object.defineProperty(exports, '__esModule', { value: true });

})));
//# sourceMappingURL=index.development.js.map
