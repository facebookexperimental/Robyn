"use strict";

Object.defineProperty(exports, "__esModule", {
    value: true
});

exports.default = function () {
    let cache = {};
    let declCache = [];

    return {
        collect(node, encoder) {
            if (node.type !== 'decl') {
                return;
            }

            if (/(grid-template|grid-template-areas)/i.test(node.prop)) {
                (0, _postcssValueParser2.default)(node.value).walk(child => {
                    if (child.type === 'string') {
                        child.value.split(/\s+/).forEach(word => {
                            if (/\.+/.test(word)) {
                                // reduce empty zones to a single `.`
                                node.value = node.value.replace(word, ".");
                            } else if (word && RESERVED_KEYWORDS.indexOf(word.toLowerCase()) === -1) {
                                (0, _cache2.default)(word, encoder, cache);
                            }
                        });
                    }
                });

                declCache.push(node);
            } else if (node.prop.toLowerCase() === 'grid-area') {
                (0, _postcssValueParser2.default)(node.value).walk(child => {
                    if (child.type === 'word' && RESERVED_KEYWORDS.indexOf(child.value) === -1) {
                        (0, _cache2.default)(child.value, encoder, cache);
                    }
                });

                declCache.push(node);
            }
        },

        transform() {
            declCache.forEach(decl => {
                decl.value = (0, _postcssValueParser2.default)(decl.value).walk(node => {
                    if (/(grid-template|grid-template-areas)/i.test(decl.prop)) {
                        node.value.split(/\s+/).forEach(word => {
                            if (word in cache) {
                                node.value = node.value.replace(word, cache[word].ident);
                            }
                        });
                        node.value = node.value.replace(/\s+/g, " "); // merge white-spaces
                    }

                    if (decl.prop.toLowerCase() === 'grid-area' && !(0, _isNum2.default)(node)) {
                        if (node.value in cache) {
                            node.value = cache[node.value].ident;
                        }
                    }

                    return false;
                }).toString();
            });

            // reset cache after transform
            declCache = [];
        }
    };
};

var _postcssValueParser = require("postcss-value-parser");

var _postcssValueParser2 = _interopRequireDefault(_postcssValueParser);

var _cache = require("./cache");

var _cache2 = _interopRequireDefault(_cache);

var _isNum = require("./isNum");

var _isNum2 = _interopRequireDefault(_isNum);

function _interopRequireDefault(obj) { return obj && obj.__esModule ? obj : { default: obj }; }

const RESERVED_KEYWORDS = ["auto", "span", "inherit", "initial", "unset"];

module.exports = exports["default"];