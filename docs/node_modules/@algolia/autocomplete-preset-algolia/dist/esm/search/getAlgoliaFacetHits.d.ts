import { SearchForFacetValuesParams } from './searchForFacetValues';
declare type FacetHit = {
    label: string;
    count: number;
    _highlightResult: {
        label: {
            value: string;
        };
    };
};
export declare function getAlgoliaFacetHits({ searchClient, queries, userAgents, }: SearchForFacetValuesParams): Promise<FacetHit[][]>;
export {};
