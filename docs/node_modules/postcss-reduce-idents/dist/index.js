'use strict';

Object.defineProperty(exports, "__esModule", {
    value: true
});

var _postcss = require('postcss');

var _postcss2 = _interopRequireDefault(_postcss);

var _encode = require('./lib/encode');

var _encode2 = _interopRequireDefault(_encode);

var _counter = require('./lib/counter');

var _counter2 = _interopRequireDefault(_counter);

var _counterStyle = require('./lib/counter-style');

var _counterStyle2 = _interopRequireDefault(_counterStyle);

var _keyframes = require('./lib/keyframes');

var _keyframes2 = _interopRequireDefault(_keyframes);

var _gridTemplate = require('./lib/grid-template');

var _gridTemplate2 = _interopRequireDefault(_gridTemplate);

function _interopRequireDefault(obj) { return obj && obj.__esModule ? obj : { default: obj }; }

exports.default = _postcss2.default.plugin('postcss-reduce-idents', ({
    counter = true,
    counterStyle = true,
    keyframes = true,
    gridTemplate = true,
    encoder = _encode2.default
} = {}) => {
    const reducers = [];

    counter && reducers.push((0, _counter2.default)());
    counterStyle && reducers.push((0, _counterStyle2.default)());
    keyframes && reducers.push((0, _keyframes2.default)());
    gridTemplate && reducers.push((0, _gridTemplate2.default)());

    return css => {
        css.walk(node => {
            reducers.forEach(reducer => reducer.collect(node, encoder));
        });

        reducers.forEach(reducer => reducer.transform());
    };
});
module.exports = exports['default'];