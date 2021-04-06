'use strict';

Object.defineProperty(exports, "__esModule", {
    value: true
});

var _has = require('has');

var _has2 = _interopRequireDefault(_has);

var _postcss = require('postcss');

var _postcssValueParser = require('postcss-value-parser');

var _postcssValueParser2 = _interopRequireDefault(_postcssValueParser);

var _cssnanoUtilSameParent = require('cssnano-util-same-parent');

var _cssnanoUtilSameParent2 = _interopRequireDefault(_cssnanoUtilSameParent);

function _interopRequireDefault(obj) { return obj && obj.__esModule ? obj : { default: obj }; }

function canonical(obj) {
    // Prevent potential infinite loops
    let stack = 50;
    return function recurse(key) {
        if ((0, _has2.default)(obj, key) && obj[key] !== key && stack) {
            stack--;
            return recurse(obj[key]);
        }
        stack = 50;
        return key;
    };
}

function mergeAtRules(css, pairs) {
    pairs.forEach(pair => {
        pair.cache = [];
        pair.replacements = [];
        pair.decls = [];
    });

    let relevant;

    css.walk(node => {
        if (node.type === 'atrule') {
            relevant = pairs.filter(pair => pair.atrule.test(node.name.toLowerCase()))[0];
            if (!relevant) {
                return;
            }
            if (relevant.cache.length < 1) {
                relevant.cache.push(node);
                return;
            } else {
                let toString = node.nodes.toString();
                relevant.cache.forEach(cached => {
                    if (cached.name.toLowerCase() === node.name.toLowerCase() && (0, _cssnanoUtilSameParent2.default)(cached, node) && cached.nodes.toString() === toString) {
                        cached.remove();
                        relevant.replacements[cached.params] = node.params;
                    }
                });
                relevant.cache.push(node);
                return;
            }
        }
        if (node.type === 'decl') {
            relevant = pairs.filter(pair => pair.decl.test(node.prop.toLowerCase()))[0];
            if (!relevant) {
                return;
            }
            relevant.decls.push(node);
        }
    });

    pairs.forEach(pair => {
        let canon = canonical(pair.replacements);
        pair.decls.forEach(decl => {
            decl.value = (0, _postcssValueParser2.default)(decl.value).walk(node => {
                if (node.type === 'word') {
                    node.value = canon(node.value);
                }
            }).toString();
        });
    });
}

exports.default = (0, _postcss.plugin)('postcss-merge-idents', () => {
    return css => {
        mergeAtRules(css, [{
            atrule: /keyframes/i,
            decl: /animation/i
        }, {
            atrule: /counter-style/i,
            decl: /(list-style|system)/i
        }]);
    };
});
module.exports = exports['default'];