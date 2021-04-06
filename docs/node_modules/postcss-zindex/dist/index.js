'use strict';

Object.defineProperty(exports, "__esModule", {
    value: true
});

var _postcss = require('postcss');

var _layerCache = require('./lib/layerCache');

var _layerCache2 = _interopRequireDefault(_layerCache);

function _interopRequireDefault(obj) { return obj && obj.__esModule ? obj : { default: obj }; }

exports.default = (0, _postcss.plugin)('postcss-zindex', (opts = {}) => {
    return css => {
        const cache = new _layerCache2.default(opts);
        const nodes = [];
        let abort = false;
        // First pass; cache all z indexes
        css.walkDecls(/z-index/i, decl => {
            // Check that no negative values exist. Rebasing is only
            // safe if all indices are positive numbers.
            if (decl.value[0] === '-') {
                abort = true;
                // Stop PostCSS iterating through the rest of the decls
                return false;
            }
            nodes.push(decl);
            cache.addValue(decl.value);
        });

        // Abort if we found any negative values
        // or there are no z-index declarations
        if (abort || !nodes.length) {
            return;
        }

        cache.optimizeValues();

        // Second pass; optimize
        nodes.forEach(decl => {
            // Need to coerce to string so that the
            // AST is updated correctly
            decl.value = cache.getValue(decl.value).toString();
        });
    };
});
module.exports = exports['default'];