import { DocSearchHit } from './DocSearchHit';
export declare type InternalDocSearchHit = DocSearchHit & {
    __docsearch_parent: null | InternalDocSearchHit;
};
