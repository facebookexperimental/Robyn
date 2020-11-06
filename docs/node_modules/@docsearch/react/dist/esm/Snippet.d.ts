/// <reference types="react" />
import { StoredDocSearchHit } from './types';
interface SnippetProps<TItem> {
    [prop: string]: unknown;
    hit: TItem;
    attribute: string;
    tagName?: string;
}
export declare function Snippet<TItem extends StoredDocSearchHit>({ hit, attribute, tagName, ...rest }: SnippetProps<TItem>): import("react").DOMElement<{
    dangerouslySetInnerHTML: {
        __html: any;
    };
}, Element>;
export {};
