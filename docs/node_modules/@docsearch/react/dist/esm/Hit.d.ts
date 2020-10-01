import React from 'react';
import { DocSearchHit } from './types';
interface HitProps {
    hit: DocSearchHit;
    children: React.ReactNode;
}
export declare function Hit({ hit, children }: HitProps): JSX.Element;
export {};
