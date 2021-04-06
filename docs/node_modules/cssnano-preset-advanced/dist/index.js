'use strict';

Object.defineProperty(exports, "__esModule", {
    value: true
});
exports.default = advancedPreset;

var _cssnanoPresetDefault = require('cssnano-preset-default');

var _cssnanoPresetDefault2 = _interopRequireDefault(_cssnanoPresetDefault);

var _postcssDiscardUnused = require('postcss-discard-unused');

var _postcssDiscardUnused2 = _interopRequireDefault(_postcssDiscardUnused);

var _postcssMergeIdents = require('postcss-merge-idents');

var _postcssMergeIdents2 = _interopRequireDefault(_postcssMergeIdents);

var _postcssReduceIdents = require('postcss-reduce-idents');

var _postcssReduceIdents2 = _interopRequireDefault(_postcssReduceIdents);

var _postcssZindex = require('postcss-zindex');

var _postcssZindex2 = _interopRequireDefault(_postcssZindex);

var _autoprefixer = require('autoprefixer');

var _autoprefixer2 = _interopRequireDefault(_autoprefixer);

function _interopRequireDefault(obj) { return obj && obj.__esModule ? obj : { default: obj }; }

const defaultOpts = {
    autoprefixer: {
        add: false
    }
};

function advancedPreset(opts = {}) {
    const options = Object.assign({}, defaultOpts, opts);

    const plugins = [...(0, _cssnanoPresetDefault2.default)(options).plugins, [_autoprefixer2.default, options.autoprefixer], [_postcssDiscardUnused2.default, options.discardUnused], [_postcssMergeIdents2.default, options.mergeIdents], [_postcssReduceIdents2.default, options.reduceIdents], [_postcssZindex2.default, options.zindex]];

    return { plugins };
}
module.exports = exports['default'];