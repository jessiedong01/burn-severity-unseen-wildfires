// ============================================================
// GEE Export Script — Camp Fire (2018), Landsat 8
// Paste this entire file into code.earthengine.google.com
// and click "Run". Two export tasks will appear in the
// Tasks panel (top-right). Click "RUN" on each to start.
// Files land in Google Drive → burn_severity/
// ============================================================

// ── Region of interest ────────────────────────────────────
var roi = ee.Geometry.Rectangle([-121.85, 39.585, -121.29, 40.035]);

Map.setCenter(-121.57, 39.81, 10);
Map.addLayer(roi, {color: 'FF0000'}, 'ROI');

// ── Cloud masking (Landsat 8 C2 L2 QA_PIXEL) ─────────────
function maskClouds(image) {
  var qa = image.select('QA_PIXEL');
  var mask = qa.bitwiseAnd(1 << 3).eq(0)
               .and(qa.bitwiseAnd(1 << 4).eq(0));
  return image.updateMask(mask);
}

// ── Build a median composite for a date window ────────────
// C02 L2 band names are SR_B2, SR_B3, ... SR_B7
var SR_BANDS  = ['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7'];
var OUT_BANDS = ['B2',    'B3',    'B4',    'B5',    'B6',    'B7'];

function getComposite(startDate, endDate) {
  return ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
    .filterBounds(roi)
    .filterDate(startDate, endDate)
    .map(maskClouds)
    .select(SR_BANDS, OUT_BANDS)
    .median()
    .clip(roi);
}

// ── Pre-fire: Aug 1 – Nov 7 2018 ─────────────────────────
var prefire  = getComposite('2018-08-01', '2018-11-07');

// ── Post-fire: Nov 9 2018 – Jan 1 2019 ───────────────────
var postfire = getComposite('2018-11-09', '2019-01-01');

// ── Quick-look: false-color (SWIR1/NIR/Red = B6/B5/B4) ───
var visParams = {bands: ['B6', 'B5', 'B4'], min: 5000, max: 20000};
Map.addLayer(prefire,  visParams, 'Pre-fire  (SWIR/NIR/Red)');
Map.addLayer(postfire, visParams, 'Post-fire (SWIR/NIR/Red)');

// ── Export tasks (no Object.assign — GEE sandbox limitation) ─
var exportOptions = {
  region: roi,
  scale: 30,
  crs: 'EPSG:4326',
  maxPixels: 1e9,
  folder: 'burn_severity',
  fileFormat: 'GeoTIFF',
};

Export.image.toDrive({
  image: prefire,
  description: 'camp_fire_pre_2018',
  fileNamePrefix: 'camp_fire_pre_2018',
  region: roi,
  scale: 30,
  maxPixels: 1e9,
  folder: 'burn_severity',
  fileFormat: 'GeoTIFF',
});

Export.image.toDrive({
  image: postfire,
  description: 'camp_fire_post_2018',
  fileNamePrefix: 'camp_fire_post_2018',
  region: roi,
  scale: 30,
  maxPixels: 1e9,
  folder: 'burn_severity',
  fileFormat: 'GeoTIFF',
});

// ── Scene counts ──────────────────────────────────────────
print('Pre-fire scene count:',
  ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
    .filterBounds(roi).filterDate('2018-08-01', '2018-11-07').size());

print('Post-fire scene count:',
  ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
    .filterBounds(roi).filterDate('2018-11-09', '2019-01-01').size());
