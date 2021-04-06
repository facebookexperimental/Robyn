"use strict";

Object.defineProperty(exports, "__esModule", {
    value: true
});

exports.default = function (value, encoder, cache) {
    if (cache[value]) {
        return;
    }

    cache[value] = {
        ident: encoder(value, Object.keys(cache).length),
        count: 0
    };
};

module.exports = exports["default"];