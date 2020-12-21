import { Matcher, MatcherConfig } from "./matcher";
import { HashtagServices } from "../autolinker";
import { Match } from "../match/match";
/**
 * @class Autolinker.matcher.Hashtag
 * @extends Autolinker.matcher.Matcher
 *
 * Matcher to find HashtagMatch matches in an input string.
 */
export declare class HashtagMatcher extends Matcher {
    /**
     * @cfg {String} serviceName
     *
     * The service to point hashtag matches to. See {@link Autolinker#hashtag}
     * for available values.
     */
    protected readonly serviceName: HashtagServices;
    /**
     * The regular expression to match Hashtags. Example match:
     *
     *     #asdf
     *
     * @protected
     * @property {RegExp} matcherRegex
     */
    protected matcherRegex: RegExp;
    /**
     * The regular expression to use to check the character before a username match to
     * make sure we didn't accidentally match an email address.
     *
     * For example, the string "asdf@asdf.com" should not match "@asdf" as a username.
     *
     * @protected
     * @property {RegExp} nonWordCharRegex
     */
    protected nonWordCharRegex: RegExp;
    /**
     * @method constructor
     * @param {Object} cfg The configuration properties for the Match instance,
     *   specified in an Object (map).
     */
    constructor(cfg: HashtagMatcherConfig);
    /**
     * @inheritdoc
     */
    parseMatches(text: string): Match[];
}
export interface HashtagMatcherConfig extends MatcherConfig {
    serviceName: HashtagServices;
}
