import { AutocompleteApi, AutocompleteOptions, BaseItem } from './types';
export declare function createAutocomplete<TItem extends BaseItem, TEvent = Event, TMouseEvent = MouseEvent, TKeyboardEvent = KeyboardEvent>(options: AutocompleteOptions<TItem>): AutocompleteApi<TItem, TEvent, TMouseEvent, TKeyboardEvent>;
