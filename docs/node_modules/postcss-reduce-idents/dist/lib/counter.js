"use strict";

Object.defineProperty(exports, "__esModule", {
    value: true
});

exports.default = function () {
    let cache = {};
    let declOneCache = [];
    let declTwoCache = [];

    return {
        collect(node, encoder) {
            const { prop, type } = node;

            if (type !== 'decl') {
                return;
            }

            if (/counter-(reset|increment)/i.test(prop)) {
                node.value = (0, _postcssValueParser2.default)(node.value).walk(child => {
                    if (child.type === 'word' && !(0, _isNum2.default)(child) && RESERVED_KEYWORDS.indexOf(child.value.toLowerCase()) === -1) {
                        (0, _cache2.default)(child.value, encoder, cache);

                        child.value = cache[child.value].ident;
                    }
                });

                declOneCache.push(node);
            } else if (/content/i.test(prop)) {
                declTwoCache.push(node);
            }
        },

        transform() {
            declTwoCache.forEach(decl => {
                decl.value = (0, _postcssValueParser2.default)(decl.value).walk(node => {
                    const { type } = node;

                    const value = node.value.toLowerCase();

                    if (type === 'function' && (value === 'counter' || value === 'counters')) {
                        (0, _postcssValueParser.walk)(node.nodes, child => {
                            if (child.type === 'word' && child.value in cache) {
                                cache[child.value].count++;

                                child.value = cache[child.value].ident;
                            }
                        });
                    }

                    if (type === 'space') {
                        node.value = ' ';
                    }

                    return false;
                }).toString();
            });

            declOneCache.forEach(decl => {
                decl.value = decl.value.walk(node => {
                    if (node.type === 'word' && !(0, _isNum2.default)(node)) {
                        Object.keys(cache).forEach(key => {
                            const cached = cache[key];

                            if (cached.ident === node.value && !cached.count) {
                                node.value = key;
                            }
                        });
                    }
                }).toString();
            });

            // reset cache after transform
            declOneCache = [];
            declTwoCache = [];
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

const RESERVED_KEYWORDS = ["unset", "initial", "inherit", "none"];

module.exports = exports["default"];