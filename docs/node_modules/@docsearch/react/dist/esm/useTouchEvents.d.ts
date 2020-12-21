import { AutocompleteApi } from '@algolia/autocomplete-core';
interface UseTouchEventsProps {
    getEnvironmentProps: AutocompleteApi<any>['getEnvironmentProps'];
    panelElement: HTMLDivElement | null;
    searchBoxElement: HTMLDivElement | null;
    inputElement: HTMLInputElement | null;
}
export declare function useTouchEvents({ getEnvironmentProps, panelElement, searchBoxElement, inputElement, }: UseTouchEventsProps): void;
export {};
