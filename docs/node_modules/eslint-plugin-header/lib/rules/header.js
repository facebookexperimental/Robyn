"use strict";

var fs = require("fs");
var commentParser = require("../comment-parser");
var os = require("os");

function isPattern(object) {
    return typeof object === "object" && object.hasOwnProperty("pattern");
}

function match(actual, expected) {
    if (expected.test) {
        return expected.test(actual);
    } else {
        return expected === actual;
    }
}

function excludeShebangs(comments) {
    return comments.filter(function(comment) {
        return comment.type !== "Shebang";
    });
}

function getLeadingComments(context, node) {
    return node.body.length ?
        context.getComments(node.body[0]).leading :
        context.getComments(node).leading;
}

function genCommentBody(commentType, textArray, eol) {
    if (commentType === "block") {
        return "/*" + textArray.join(eol) + "*/" + eol;
    } else {
        return "//" + textArray.join(eol + "//") + eol;
    }
}

function genCommentsRange(context, comments, eol) {
    var start = comments[0].range[0];
    var end = comments.slice(-1)[0].range[1];
    if (context.getSourceCode().text[end] === eol) {
        end += eol.length;
    }
    return [start, end];
}

function genPrependFixer(commentType, node, headerLines, eol) {
    return function(fixer) {
        return fixer.insertTextBefore(
            node,
            genCommentBody(commentType, headerLines, eol)
        );
    };
}

function genReplaceFixer(commentType, context, leadingComments, headerLines, eol) {
    return function(fixer) {
        return fixer.replaceTextRange(
            genCommentsRange(context, leadingComments, eol),
            genCommentBody(commentType, headerLines, eol)
        );
    };
}

function findSettings(options) {
    var lastOption = options.length > 0 ? options[options.length - 1] : null;
    if (typeof lastOption === "object" && !Array.isArray(lastOption) && lastOption !== null && !lastOption.hasOwnProperty("pattern")) {
        return lastOption;
    }
    return null;
}

function getEOL(options) {
    var settings = findSettings(options);
    if (settings && settings.lineEndings === "unix") {
        return "\n";
    }
    if (settings && settings.lineEndings === "windows") {
        return "\r\n";
    }
    return os.EOL;
}

module.exports = function(context) {
    var options = context.options;

    var eol = getEOL(options);

    // If just one option then read comment from file
    if (options.length === 1 || (options.length === 2 && findSettings(options))) {
        var text = fs.readFileSync(context.options[0], "utf8");
        options = commentParser(text);
    }

    var commentType = options[0];
    var headerLines, fixLines = [];
    // If any of the lines are regular expressions, then we can't
    // automatically fix them. We set this to true below once we
    // ensure none of the lines are of type RegExp
    var canFix = false;
    if (Array.isArray(options[1])) {
        canFix = true;
        headerLines = options[1].map(function(line) {
            var isRegex = isPattern(line);
            // Can only fix regex option if a template is also provided
            if (isRegex && !line.template) {
                canFix = false;
            }
            fixLines.push(line.template || line);
            return isRegex ? new RegExp(line.pattern) : line;
        });
    } else if (isPattern(options[1])) {
        var line = options[1];
        headerLines = [new RegExp(line.pattern)];
        fixLines.push(line.template || line);
        // Same as above for regex and template
        canFix = !!line.template;
    } else {
        canFix = true;
        headerLines = options[1].split(/\r?\n/);
        fixLines = headerLines;
    }

    return {
        Program: function(node) {
            var leadingComments = excludeShebangs(getLeadingComments(context, node));

            if (!leadingComments.length) {
                context.report({
                    loc: node.loc,
                    message: "missing header",
                    fix: canFix ? genPrependFixer(commentType, node, fixLines, eol) : null
                });
            } else if (leadingComments[0].type.toLowerCase() !== commentType) {
                context.report({
                    loc: node.loc,
                    message: "header should be a {{commentType}} comment",
                    data: {
                        commentType: commentType
                    },
                    fix: canFix ? genReplaceFixer(commentType, context, leadingComments, fixLines, eol) : null
                });
            } else {
                if (commentType === "line") {
                    if (leadingComments.length < headerLines.length) {
                        context.report({
                            loc: node.loc,
                            message: "incorrect header",
                            fix: canFix ? genReplaceFixer(commentType, context, leadingComments, fixLines, eol) : null
                        });
                        return;
                    }
                    for (var i = 0; i < headerLines.length; i++) {
                        if (!match(leadingComments[i].value, headerLines[i])) {
                            context.report({
                                loc: node.loc,
                                message: "incorrect header",
                                fix: canFix ? genReplaceFixer(commentType, context, leadingComments, fixLines, eol) : null
                            });
                            return;
                        }
                    }
                } else {
                    // if block comment pattern has more than 1 line, we also split the comment
                    var leadingLines = [leadingComments[0].value];
                    if (headerLines.length > 1) {
                        leadingLines = leadingComments[0].value.split(/\r?\n/);
                    }

                    var hasError = false;
                    if (leadingLines.length > headerLines.length) {
                        hasError = true;
                    }
                    for (i = 0; !hasError && i < headerLines.length; i++) {
                        if (!match(leadingLines[i], headerLines[i])) {
                            hasError = true;
                        }
                    }

                    if (hasError) {
                        if (canFix && headerLines.length > 1) {
                            fixLines = [fixLines.join(eol)];
                        }
                        context.report({
                            loc: node.loc,
                            message: "incorrect header",
                            fix: canFix ? genReplaceFixer(commentType, context, leadingComments, fixLines, eol) : null
                        });
                    }
                }
            }
        }
    };
};
