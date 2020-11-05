"use strict";

// This is a really simple and dumb parser, that looks just for a
// single kind of comment. It won't detect multiple block comments.

module.exports = function commentParser(text) {
    text = text.trim();

    if (text.substr(0, 2) === "//") {
        return [
            "line",
            text.split(/\r?\n/).map(function(line) {
                return line.substr(2);
            })
        ];
    } else if (
        text.substr(0, 2) === "/*" &&
        text.substr(-2) === "*/"
    ) {
        return ["block", text.substring(2, text.length - 2)];
    } else {
        throw new Error("Could not parse comment file: the file must contain either just line comments (//) or a single block comment (/* ... */)");
    }
};
