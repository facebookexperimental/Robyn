declare global {
    const __webpack_require__: {
        gca: (name: string) => string;
    };
    interface Navigator {
        connection: any;
    }
}
declare const docusaurus: {
    prefetch: (routePath: string) => boolean;
    preload: (routePath: string) => boolean;
};
export default docusaurus;
//# sourceMappingURL=docusaurus.d.ts.map