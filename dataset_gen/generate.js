#!/usr/bin/env node
'use strict';

/**
 * Generates a labeled Sudoku board image database by driving the real
 * game (game-library/games/sudoku) in a headless browser. Each sample:
 *   - picks a random difficulty, background gradient, cell font size, theme
 *     (dark/light), high-contrast mode, and font family
 *   - asks the game's own generator/solver for a puzzle
 *   - screenshots the rendered #board element
 *   - records the 81-cell grid (0 = blank) as the label
 *
 * Usage:
 *   node generate.js --count 20000 --out ./output [options]
 *
 * Options:
 *   --count <n>         Number of images to generate (required)
 *   --out <dir>          Output directory (default: ./output)
 *   --game <path>        Path to the sudoku index.html (default: ../game-library/games/sudoku/index.html)
 *   --difficulty <d>     easy|medium|hard|expert|mixed (default: mixed)
 *   --concurrency <n>    Parallel browser pages (default: 4)
 *   --min-font <px>      Minimum cell font size in px (default: 14)
 *   --max-font <px>      Maximum cell font size in px (default: 34)
 *   --min-border <px>    Minimum grid line width in px (default: 1)
 *   --max-border <px>    Maximum grid line width in px (default: 4)
 *   --max-rotation <deg> Max +/- rotation applied to the board (default: 4)
 *   --min-opacity <0-1>  Minimum board opacity (default: 0.85)
 *   --max-opacity <0-1>  Maximum board opacity (default: 1)
 *   --format <fmt>       png|jpeg (default: png)
 *   --resume             Skip samples that already exist in --out
 */

const fs = require('fs');
const path = require('path');
const { chromium } = require('playwright');

function parseArgs(argv) {
  const args = {
    count: null,
    out: path.join(__dirname, 'output'),
    game: path.join(__dirname, '..', 'game-library', 'games', 'sudoku', 'index.html'),
    difficulty: 'mixed',
    concurrency: 4,
    minFont: 14,
    maxFont: 34,
    minBorder: 1,
    maxBorder: 4,
    maxRotation: 4,
    minOpacity: 0.85,
    maxOpacity: 1,
    format: 'png',
    resume: false
  };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    const next = () => argv[++i];
    switch (a) {
      case '--count': args.count = Number(next()); break;
      case '--out': args.out = path.resolve(next()); break;
      case '--game': args.game = path.resolve(next()); break;
      case '--difficulty': args.difficulty = next(); break;
      case '--concurrency': args.concurrency = Number(next()); break;
      case '--min-font': args.minFont = Number(next()); break;
      case '--max-font': args.maxFont = Number(next()); break;
      case '--min-border': args.minBorder = Number(next()); break;
      case '--max-border': args.maxBorder = Number(next()); break;
      case '--max-rotation': args.maxRotation = Number(next()); break;
      case '--min-opacity': args.minOpacity = Number(next()); break;
      case '--max-opacity': args.maxOpacity = Number(next()); break;
      case '--format': args.format = next(); break;
      case '--resume': args.resume = true; break;
      case '--help': case '-h':
        console.log(fs.readFileSync(__filename, 'utf8').split('/**')[1].split('*/')[0]
          .split('\n').map(l => l.replace(/^\s*\* ?/, '')).join('\n'));
        process.exit(0);
        break;
      default:
        console.error(`Unknown argument: ${a}`);
        process.exit(1);
    }
  }
  if (!args.count || args.count <= 0) {
    console.error('Missing required --count <n>. Use --help for usage.');
    process.exit(1);
  }
  if (!fs.existsSync(args.game)) {
    console.error(`Game file not found: ${args.game}`);
    process.exit(1);
  }
  return args;
}

const DIFFICULTIES = ['easy', 'medium', 'hard', 'expert'];
const THEMES = ['dark', 'light'];
const HIGH_CONTRAST_CHANCE = 0.15;
// Generic families always resolve to *some* installed font, so glyph shapes
// actually vary across samples instead of every image sharing one font.
const FONT_STACKS = [
  '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif',
  'serif',
  'monospace',
  'sans-serif',
  'cursive',
];

function randInt(min, max) { return Math.floor(Math.random() * (max - min + 1)) + min; }

function randFloat(min, max) { return Math.random() * (max - min) + min; }

function randomHexColor() {
  const r = randInt(0, 255), g = randInt(0, 255), b = randInt(0, 255);
  return '#' + [r, g, b].map(v => v.toString(16).padStart(2, '0')).join('');
}

function pickDifficulty(requested) {
  if (requested === 'mixed') return DIFFICULTIES[randInt(0, DIFFICULTIES.length - 1)];
  return requested;
}

async function generateOne(page, args, index, padWidth, labelStream) {
  const difficulty = pickDifficulty(args.difficulty);
  const fontSize = randInt(args.minFont, args.maxFont);
  const borderWidth = randFloat(args.minBorder, args.maxBorder);
  const rotation = randFloat(-args.maxRotation, args.maxRotation);
  const opacity = randFloat(args.minOpacity, args.maxOpacity);
  const bg1 = randomHexColor();
  const bg2 = randomHexColor();
  const theme = THEMES[randInt(0, THEMES.length - 1)];
  const highContrast = Math.random() < HIGH_CONTRAST_CHANCE;
  const font = FONT_STACKS[randInt(0, FONT_STACKS.length - 1)];

  const grid = await page.evaluate(({ difficulty, fontSize, borderWidth, rotation, opacity, bg1, bg2, theme, highContrast, font }) => {
    const root = document.documentElement;
    root.style.setProperty('--cell-font-size', fontSize + 'px');
    root.style.setProperty('--grid-border-width', borderWidth + 'px');
    root.style.setProperty('--grid-border-width-strong', (borderWidth * 2.2) + 'px');
    root.style.setProperty('--board-transform', `rotate(${rotation}deg)`);
    root.style.setProperty('--board-opacity', String(opacity));
    root.style.setProperty('--bg-grad-1', bg1);
    root.style.setProperty('--bg-grad-2', bg2);
    root.style.setProperty('--font', font);
    root.setAttribute('data-theme', theme);
    document.body.classList.toggle('high-contrast', highContrast);
    window.__sudokuApi.Game.newGame(difficulty);
    return window.__sudokuApi.state.puzzle.slice();
  }, { difficulty, fontSize, borderWidth, rotation, opacity, bg1, bg2, theme, highContrast, font });

  // Let the browser paint the new board/background before capturing it.
  await page.waitForTimeout(30);

  const ext = args.format === 'jpeg' ? 'jpg' : 'png';
  const fileName = `sample_${String(index).padStart(padWidth, '0')}.${ext}`;

  // Rotation grows the board's on-screen bounding box; capture that box plus a
  // margin (via a page-level clip, not an element screenshot) so the tilted
  // corners show surrounding background instead of getting cut off.
  const board = page.locator('#board');
  const box = await board.boundingBox();
  const margin = 24;
  const viewport = page.viewportSize();
  const clip = {
    x: Math.max(0, box.x - margin),
    y: Math.max(0, box.y - margin),
    width: Math.min(viewport.width, box.width + margin * 2),
    height: Math.min(viewport.height, box.height + margin * 2)
  };
  await page.screenshot({ path: path.join(args.out, 'images', fileName), type: args.format, clip });

  const record = {
    file: fileName,
    grid: grid.join(''),
    difficulty,
    fontSizePx: fontSize,
    borderWidthPx: Math.round(borderWidth * 100) / 100,
    rotationDeg: Math.round(rotation * 100) / 100,
    opacity: Math.round(opacity * 100) / 100,
    bgGrad1: bg1,
    bgGrad2: bg2,
    theme,
    highContrast,
    font
  };
  labelStream.write(JSON.stringify(record) + '\n');
}

async function worker(id, browser, args, queue, padWidth, labelStream, progress) {
  const page = await browser.newPage({ viewport: { width: 900, height: 900 } });
  await page.goto('file://' + args.game);
  await page.waitForFunction(() => !!window.__sudokuApi);
  await page.evaluate(() => document.body.classList.add('capture-mode'));

  while (queue.length) {
    const index = queue.pop();
    if (index === undefined) break;
    try {
      await generateOne(page, args, index, padWidth, labelStream);
      progress.done++;
      if (progress.done % 50 === 0 || progress.done === progress.total) {
        process.stdout.write(`\r${progress.done}/${progress.total} images generated`);
      }
    } catch (err) {
      console.error(`\nSample ${index} failed:`, err.message);
    }
  }
  await page.close();
}

async function main() {
  const args = parseArgs(process.argv.slice(2));

  fs.mkdirSync(path.join(args.out, 'images'), { recursive: true });
  const labelsPath = path.join(args.out, 'labels.jsonl');

  const padWidth = String(args.count).length;
  let startIndex = 0;
  if (args.resume && fs.existsSync(labelsPath)) {
    const existing = fs.readFileSync(labelsPath, 'utf8').split('\n').filter(Boolean).length;
    startIndex = existing;
    console.log(`Resuming: ${existing} samples already present, continuing from index ${startIndex}.`);
  } else if (!args.resume) {
    fs.writeFileSync(labelsPath, '');
  }

  const queue = [];
  for (let i = args.count - 1; i >= startIndex; i--) queue.push(i);

  const labelStream = fs.createWriteStream(labelsPath, { flags: 'a' });
  const progress = { done: 0, total: queue.length };

  console.log(`Generating ${queue.length} images -> ${args.out}`);
  console.log(`Difficulty: ${args.difficulty} | font ${args.minFont}-${args.maxFont}px | concurrency ${args.concurrency}`);

  const browser = await chromium.launch();
  const workers = [];
  for (let w = 0; w < args.concurrency; w++) {
    workers.push(worker(w, browser, args, queue, padWidth, labelStream, progress));
  }
  await Promise.all(workers);
  await browser.close();
  labelStream.end();

  console.log(`\nDone. ${progress.done} images written to ${path.join(args.out, 'images')}`);
  console.log(`Labels: ${labelsPath}`);
}

main().catch(err => { console.error(err); process.exit(1); });
