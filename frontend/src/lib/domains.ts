/**
 * Copy and teaching examples that follow the served artifact.
 *
 * The backend reports `corpus` on `GET /model` -- the stem of the training
 * file the artifact was actually fitted on -- and this maps it to the words
 * the UI uses and the examples it offers. Without that, serving
 * `absa-rest-v1` would show restaurant metrics on the Performance page and
 * then hand the user "Battery life is amazing but the screen is dim." to try,
 * which the restaurant model answers confidently and wrongly.
 *
 * Matching is on a substring of the stem, not an exact filename, because the
 * same corpus arrives under several names (`Laptop_Train_v2`,
 * `Laptops_Test_Gold`). An unrecognised corpus falls back to `null`: generic
 * copy and *no* examples, since an example has to be a real sentence whose
 * aspect is a literal span of itself, and there is no way to invent one for a
 * corpus this file has never heard of.
 */

export interface Example {
  label: string
  hint: string
  text: string
  aspect: string
}

export interface Domain {
  /** Plural noun for the review subject, lower case: "laptop reviews". */
  subject: string
  /** The corpus, named as a citation for the footer. */
  corpusName: string
  examples: readonly Example[]
}

const LAPTOPS: Domain = {
  subject: 'Laptop reviews',
  corpusName: 'the SemEval-2014 Task 4 laptop corpus',
  examples: [
    {
      label: 'Opposite in one sentence',
      hint: 'Analyze “screen”, then “Battery life”, and compare',
      text: 'Battery life is amazing but the screen is dim.',
      aspect: 'screen',
    },
    {
      label: 'Three aspects',
      hint: 'One review, three verdicts',
      text: 'The keyboard feels great, but the trackpad is unresponsive and the price is too high.',
      aspect: 'trackpad',
    },
    {
      label: 'Mentioned, not judged',
      hint: 'What “neutral” actually means here',
      text: 'It comes with a 500GB hard drive and a six-cell battery.',
      aspect: 'hard drive',
    },
    {
      label: 'An aspect that is absent',
      hint: 'Refused, because the model cannot abstain',
      text: 'The battery life is great but the screen is dim.',
      aspect: 'keyboard',
    },
  ],
}

/**
 * Every restaurant example is a verbatim SemEval-2014 training sentence,
 * original spelling included, so each aspect really is a span of its sentence
 * -- the regime the model was fitted in.
 */
const RESTAURANTS: Domain = {
  subject: 'Restaurant reviews',
  corpusName: 'the SemEval-2014 Task 4 restaurant corpus',
  examples: [
    {
      label: 'Opposite in one sentence',
      hint: 'Analyze “tables”, then “service”, and compare',
      text: 'Although the tables may be closely situated, the candle-light, food-quality and service overcompensate.',
      aspect: 'tables',
    },
    {
      label: 'Five aspects',
      hint: 'One review, three liked and two not',
      text: 'I like the somosas, chai, and the chole, but the dhosas and dhal were kinda dissapointing.',
      aspect: 'dhosas',
    },
    {
      label: 'Mentioned, not judged',
      hint: 'What “neutral” actually means here',
      text: 'And it all comes at a very reasonable price (congee, noodles, and rice dishes are no more than $3-6 each).',
      aspect: 'congee',
    },
    {
      label: 'An aspect that is absent',
      hint: 'Refused, because the model cannot abstain',
      text: 'Great wine selection, Gigondas is worth the price, and the house champagne is a great value.',
      aspect: 'service',
    },
  ],
}

/**
 * MAMS is built from restaurant reviews, so it inherits that domain's
 * examples; only the citation differs. Every MAMS sentence carries at least
 * two aspects with *different* polarities, which is the one corpus where the
 * mixed-polarity callout fires on essentially every sentence.
 */
const MAMS: Domain = {
  ...RESTAURANTS,
  corpusName: 'the MAMS-ATSA corpus (multi-aspect, multi-sentiment)',
}

/** Substring of the corpus stem, lower-cased -> domain. First match wins. */
const BY_CORPUS: ReadonlyArray<readonly [string, Domain]> = [
  ['laptop', LAPTOPS],
  ['restaurant', RESTAURANTS],
  ['mams', MAMS],
]

/** The domain for a reported corpus stem, or null if it is unrecognised. */
export function domainFor(corpus: string | null | undefined): Domain | null {
  if (!corpus) return null
  const stem = corpus.toLowerCase()
  return BY_CORPUS.find(([needle]) => stem.includes(needle))?.[1] ?? null
}
