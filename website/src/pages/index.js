/**
 * Copyright (c) Meta Platforms, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * @format
 */
import React from 'react';
import clsx from 'clsx';
import Layout from '@theme/Layout';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import useBaseUrl from '@docusaurus/useBaseUrl';
import styles from './styles.module.css';


const features = [
  {
    title: <>Reduces human bias</>,
    imageUrl: 'img/take_control.svg',
    description: (
      <>
        <ul>
          <li>Automated hyperparameter optimization with evolutionary algorithms from Facebook's AI library <a href="https://facebookresearch.github.io/nevergrad">Nevergrad</a></li>
          <li>Ridge regression in order to regularize multi-collinearity and prevent overfitting</li>
          <li>Facebook's <a href="https://facebook.github.io/prophet/">Prophet</a> library to automatically decompose the trend, seasonality and holidays patterns</li>
        </ul>
      </>
    ),
  },
  {
    title: <>Aligns with the ground-truth</>,
    imageUrl: 'img/calibrate.svg',
    description: (
      <>
        <ul>
          <li>
            It calibrates models based on ground-truth methodologies (Geo-based, Facebook lift, MTA, etc.)
          </li>
          <li>
            Facebook <a href="https://facebookresearch.github.io/nevergrad">Nevergrad</a>'s multi-objective optimization minimizing the error between MMM prediction and ground-truth
          </li>
        </ul>
      </>
    ),
  },
  {
    title: <>Enables actionable decision making</>,
    imageUrl: 'img/focus_on_what_matters.svg',
    description: (
      <>
        <ul>
          <li>
            Budget allocator using a gradient-based constrained non-linear solver to maximize the outcome by reallocating budgets
          </li>
          <li>
            Enables frequent modeling outcomes due to stronger automation
          </li>
          <li>
            Allows intuitive model comparisons via automatically generated model one-pagers
          </li>
        </ul>
      </>
    ),
  },
  {
    title: <>Private by Design</>,
    imageUrl: 'img/security.svg',
    description: (
      <>
        <ul>
          <li>
            Privacy friendly, with no requirement for PII or Individual log level
            data
          </li>
          <li>Not dependent on Cookies or Pixel data</li>
        </ul>
      </>
    ),
  },
];

function Feature({ imageUrl, title, description }) {
  const imgUrl = useBaseUrl(imageUrl);
  return (
    <div className={clsx('col col--6', styles.feature)}>
      {imgUrl && (
        <div className="text--center">
          <img className={styles.featureImage} src={imgUrl} alt={title} />
        </div>
      )}
      <h3>{title}</h3>
      <p className={styles.descriptionSectionText}>{description}</p>
    </div>
  );
}

function Home() {
  const context = useDocusaurusContext();
  const { siteConfig = {} } = context;

  return (
    <Layout
      description={siteConfig.tagline}
    >
      <header className={clsx('hero hero--primary', styles.heroBanner)}>
        <div className="container">
          <h1 className={clsx('hero__title', styles.heroTitle)}>{siteConfig.title}</h1>
          <p className={clsx('hero__subtitle', styles.heroSubtitle)}>Robyn is an experimental, ML-powered and semi-automated Marketing Mix Modeling (MMM) open source package.</p>
          <div className={styles.buttons}>
            <Link
              className={clsx(
                'button button--secondary button--lg'
              )}
              to={useBaseUrl('docs/quick-start/')}
            >
              Install Robyn
            </Link>
          </div>
        </div>
      </header>
      <main>
        <div className="padding-vert--xl">
          <div className="container">
            <div className="row">
              <div className={clsx('col col--6', styles.descriptionSection)}>
                <h2>A New Generation of Marketing Mix Modeling</h2>
                <p className={styles.descriptionSectionText}>{siteConfig.tagline}</p>
              </div>
              <div className="col col--6">
                <iframe
                  width="100%"
                  height="400"
                  src="https://www.youtube.com/embed/8SyKRpsXn44"
                  frameborder="0"
                  allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                  allowfullscreen
                />
              </div>
            </div>
          </div>
        </div>
          {features && features.length > 0 && (
            <section className={clsx('padding-vert--xl', styles.features)}>
              <div className="container">
                <div className="row">
                  {features.map(({ title, imageUrl, description }) => (
                    <Feature
                      key={title}
                      title={title}
                      imageUrl={imageUrl}
                      description={description}
                    />
                  ))}
                </div>
              </div>
            </section>
          )}
        <div className="container padding-vert--xl">
          <div className="row">
            <div className="col col--6 col--offset-3 text--center">
              <h2>Robyn Code Walkthrough Video</h2>
              <p className={styles.descriptionSectionText}>Please watch this walkthrough video to understand better how the code works</p>
              <iframe
                title="Robyn walkthrough video"
                width="560"
                height="315"
                src="https://www.youtube.com/embed/aIiadcfL4uw"
                frameborder="0"
                allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                allowfullscreen
              />
            </div>
          </div>
        </div>
      </main>
    </Layout>
  );
}

export default Home;
