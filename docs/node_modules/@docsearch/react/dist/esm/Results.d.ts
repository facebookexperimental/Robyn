import { AutocompleteApi, AutocompleteState } from '@francoischalifour/autocomplete-core';
import React from 'react';
import { DocSearchProps } from './DocSearch';
import { StoredDocSearchHit } from './types';
interface ResultsProps<TItem> extends AutocompleteApi<TItem, React.FormEvent, React.MouseEvent, React.KeyboardEvent> {
    title: string;
    suggestion: AutocompleteState<TItem>['suggestions'][0];
    renderIcon(props: {
        item: TItem;
        index: number;
    }): React.ReactNode;
    renderAction(props: {
        item: TItem;
        runDeleteTransition: (cb: () => void) => void;
        runFavoriteTransition: (cb: () => void) => void;
    }): React.ReactNode;
    onItemClick(item: TItem): void;
    hitComponent: DocSearchProps['hitComponent'];
}
export declare function Results<TItem extends StoredDocSearchHit>(props: ResultsProps<TItem>): JSX.Element | null;
export {};
