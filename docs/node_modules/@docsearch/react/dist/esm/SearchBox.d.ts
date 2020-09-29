import { AutocompleteApi, AutocompleteState } from '@francoischalifour/autocomplete-core';
import React, { MutableRefObject } from 'react';
import { InternalDocSearchHit } from './types';
interface SearchBoxProps extends AutocompleteApi<InternalDocSearchHit, React.FormEvent, React.MouseEvent, React.KeyboardEvent> {
    state: AutocompleteState<InternalDocSearchHit>;
    autoFocus: boolean;
    inputRef: MutableRefObject<HTMLInputElement | null>;
    onClose(): void;
}
export declare function SearchBox(props: SearchBoxProps): JSX.Element;
export {};
