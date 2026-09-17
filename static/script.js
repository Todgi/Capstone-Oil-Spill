// Variabel global untuk menyimpan referensi ke objek peta dan layer
window.map = null;
let referenceRaster = null;
let segmentationLayer = null;
let madosLayer = null;
let rasterBounds = null;
let isSplitView = false;
let map2 = null;
let layerControl = null;
let oilDriftLayer = null;
let oilDriftPath = null;
let detectedOilCoordinates = [];
let madosOilCoordinates = [];

// Tambahkan variabel global untuk animasi
var timeSliderInterval = null;
var currentTimeIndex = 0;

// Simpan referensi layer Oil Boom agar bisa dikontrol
window.oilBoomLayers = {};

// Tambahkan variabel global untuk layer optic
let opticLayer = null;

// Tambahkan objek global untuk referensi layer agar layer control berfungsi
document.addEventListener('DOMContentLoaded', function() {
    window.layers = {
        raster: null,
        segmentation: null,
        optic: null,
        mados: null
    };
});

// Inisialisasi peta saat dokumen dimuat
document.addEventListener('DOMContentLoaded', function() {
    console.log('DOM loaded, initializing map...');
    initMap();
    
    // Load oil types untuk OpenOil
    loadOilTypes();
    
    // Initialize layer controls
    initializeLayerControls();
    
    setTimeout(() => {
        if (window.map) {
            addOilBoomLayers();
            loadAOILayer();
        }
    }, 2000);
});

// Fungsi untuk inisialisasi peta
function initMap() {
    // Initialize map
    window.map = L.map('map', {
        center: [0, 0],
        zoom: 2,
        minZoom: 2,
        maxZoom: 19,
        zoomControl: true
    });

    // Define base layers
    const osmLayer = L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '© OpenStreetMap contributors'
    });

    const googleSatLayer = L.tileLayer('http://{s}.google.com/vt/lyrs=s&x={x}&y={y}&z={z}', {
        maxZoom: 20,
        subdomains: ['mt0', 'mt1', 'mt2', 'mt3'],
        attribution: '© Google'
    });

    // Add base layers to map
    const baseLayers = {
        "OpenStreetMap": osmLayer,
        "Google Satellite": googleSatLayer
    };

    // Add layer control
    window.layerControl = L.control.layers(baseLayers, null, {
        position: 'topright',
        collapsed: true // Enable hide/show mode
    }).addTo(window.map);

    // Set default base layer
    osmLayer.addTo(window.map);

    // Initialize layer variables
    window.referenceRaster = null;
    window.segmentationLayer = null;
    window.opticLayer = null;
    window.madosLayer = null;
    window.madosOilCoordinates = null;

    // Initialize layer controls
    const layerControls = {
        'raster': { button: 'rasterToggle', opacity: 'rasterOpacity', showText: 'Show SAR', hideText: 'Hide SAR' },
        'segmentation': { button: 'segmentationToggle', opacity: 'segmentationOpacity', showText: 'Show SAR Unet Segmentation', hideText: 'Hide SAR Unet Segmentation' },
        'optic': { button: 'opticToggle', opacity: 'opticOpacity', showText: 'Show Optic', hideText: 'Hide Optic' },
        'mados': { button: 'madosToggle', opacity: 'MariNeXtOpacity', showText: 'Show MariNeXt Detection', hideText: 'Hide MariNeXt Detection' }
    };

    // Set initial button states
    Object.entries(layerControls).forEach(([layerType, controls]) => {
        const button = document.getElementById(controls.button);
        const opacityControl = document.getElementById(controls.opacity);
        
        if (button) {
            button.classList.remove('active');
            button.textContent = controls.showText;
        }
        
        if (opacityControl) {
            opacityControl.disabled = true;
        }
    });

    // Add click handler for map
    window.map.on('click', function(e) {
        const lat = e.latlng.lat;
        const lng = e.latlng.lng;
        
        // Update coordinate inputs
        const latInput = document.getElementById('lat');
        const lonInput = document.getElementById('lon');
        if (latInput) latInput.value = lat.toFixed(6);
        if (lonInput) lonInput.value = lng.toFixed(6);
        
        // Update OpenOil simulation coordinates if enabled
        const useDetectedOilCheckbox = document.getElementById('useDetectedOil');
        if (useDetectedOilCheckbox && !useDetectedOilCheckbox.checked) {
            const latitudeInput = document.getElementById('latitude');
            const longitudeInput = document.getElementById('longitude');
            if (latitudeInput) latitudeInput.value = lat.toFixed(6);
            if (longitudeInput) longitudeInput.value = lng.toFixed(6);
        }
    });
}

// Fungsi untuk memproses SAR dengan U-Net
async function processSAR() {
    const sarInput = document.getElementById('sarInput');
    
    if (!sarInput.files[0]) {
        alert('Please select a SAR file');
        return;
    }

    showLoadingOverlay('Processing SAR image with U-Net...');

    const formData = new FormData();
    formData.append('file', sarInput.files[0]);

    try {
        document.getElementById('originalSARImage').style.display = 'none';
        document.getElementById('sarSegmentationImage').style.display = 'none';
        
        const response = await fetch('http://localhost:8000/predict_from_raster', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`Server responded with status ${response.status}: ${errorText}`);
        }

        const result = await response.json();
        console.log('Received SAR prediction data:', result);
        
        if (!result.success) {
            throw new Error('SAR prediction failed');
        }
        
        if (!result.original_image || !result.segmentation_image || !result.bounds) {
            throw new Error('Missing required data in response');
        }
        
        rasterBounds = L.latLngBounds(
            [result.bounds.south, result.bounds.west],
            [result.bounds.north, result.bounds.east]
        );
        
        // Remove existing layers
        if (referenceRaster) {
            window.map.removeLayer(referenceRaster);
        }
        if (segmentationLayer) {
            window.map.removeLayer(segmentationLayer);
        }
        
        // Create and add the reference raster layer
        referenceRaster = L.imageOverlay(
            `data:image/png;base64,${result.original_image}`,
            rasterBounds,
            {
                opacity: 1.0,
                className: 'original-overlay',
                interactive: true,
                zIndex: 500
            }
        ).addTo(window.map);
        window.layers.raster = referenceRaster;
        enableOpacityControl('raster', 100);

        // Create and add the segmentation layer with correct colors
        segmentationLayer = L.imageOverlay(
            `data:image/png;base64,${result.segmentation_image}`,
            rasterBounds,
            {
                opacity: 0.7,
                className: 'segmentation-overlay',
                interactive: true,
                zIndex: 1000
            }
        ).addTo(window.map);
        window.layers.segmentation = segmentationLayer;
        enableOpacityControl('segmentation', 70);
        
        // Display images
        const originalImage = document.getElementById('originalSARImage');
        originalImage.src = result.original_image.startsWith('data:image/png;base64,') 
            ? result.original_image 
            : `data:image/png;base64,${result.original_image}`;
        originalImage.style.display = 'block';
        
        const segmentationImage = document.getElementById('sarSegmentationImage');
        segmentationImage.src = result.segmentation_image.startsWith('data:image/png;base64,') 
            ? result.segmentation_image 
            : `data:image/png;base64,${result.segmentation_image}`;
        segmentationImage.style.display = 'block';
        
        window.map.fitBounds(rasterBounds);
        
        // Process oil coordinates
        if (result.oil_coordinates && result.oil_coordinates.length > 0) {
            detectedOilCoordinates = result.oil_coordinates;
            
            const useDetectedOilCheckbox = document.getElementById('useDetectedOil');
            useDetectedOilCheckbox.disabled = false;
            useDetectedOilCheckbox.checked = true;
            
            const detectedOilInfo = document.getElementById('detectedOilInfo');
            if (detectedOilInfo) {
                detectedOilInfo.innerHTML = `<strong>SAR U-Net:</strong> ${detectedOilCoordinates.length} oil points detected`;
                detectedOilInfo.style.display = 'block';
            }
            
            console.log('SAR oil coordinates detected:', detectedOilCoordinates);
        }
        
    } catch (error) {
        console.error('Error processing SAR:', error);
        alert('Failed to process SAR image: ' + error.message);
    } finally {
        hideLoadingOverlay();
    }
}

// Fungsi untuk memproses Optical dengan MADOS
async function processOptic(usePreview = false) {
    const opticInput = document.getElementById('opticInput');
    const files = opticInput.files;
    
    if (!files.length) {
        alert('Please select optical image files (.tif 11 bands or .zip)');
        return;
    }
    
    const formData = new FormData();
    
    // If ZIP, just send the first file
    if (files.length === 1 && files[0].name.toLowerCase().endsWith('.zip')) {
        formData.append('file', files[0]);
    } 
    // If multiple TIFF, check count and names
    else if (files.length >= 1 && files.length <= 11) {
        // Check if all required bands are present
        const allowedBands = [
            'B01.tif', 'B02.tif', 'B03.tif', 'B04.tif', 'B05.tif', 
            'B06.tif', 'B07.tif', 'B08.tif', 'B8A.tif', 'B11.tif', 'B12.tif'
        ];
        
        const fileNames = Array.from(files).map(f => f.name.toUpperCase());
        
        // Hanya peringatkan jika kurang dari 11 band
        if (files.length < 11) {
            console.warn('Not all 11 bands present. Will process with available bands.');
        }
        
        // Append all files to formData
        for (const file of files) {
            formData.append('files', file);
        }
    } else {
        alert('Please select either a ZIP file or up to 11 band TIFF files (B01.tif, ..., B11.tif, B8A.tif)');
        return;
    }
    
    // Add preview flag if requested
    if (usePreview) {
        formData.append('preview', 'true');
    }
    
    try {
        // Hide existing images
        document.getElementById('originalOpticImage').style.display = 'none';
        document.getElementById('madosSegmentationImage').style.display = 'none';
        
        // Show loading message
        showLoadingOverlay(usePreview ? 'Creating quick preview with MADOS...' : 'Processing optical image with MADOS...');
        
        // Process with progress updates
        const result = await processMadosProgressively(formData);
        
        console.log('MADOS result:', result);
        
        // Show detection info
        const detectionInfo = document.getElementById('madosDetectionInfo');
        const detectionText = document.getElementById('madosDetectionText');
        detectionInfo.style.display = 'block';
        
        if (result.error_band) {
            detectionText.innerHTML = `<strong>Error:</strong> ${result.error_band}`;
            return;
        }
        
        // Display segmentation image dan original image
        tampilkanOpticResult(result.original_image, result.segmentation_image);
        
        // Handle oil coordinates
        if (result.oil_coordinates && result.oil_coordinates.length > 0) {
            window.madosOilCoordinates = result.oil_coordinates;
            detectionText.innerHTML = `<strong>MADOS Detection:</strong> ${result.oil_coordinates.length} oil points detected`;
            
            // Enable "Use detected oil" checkbox for OpenOil simulation
            const useDetectedOilCheckbox = document.getElementById('useDetectedOil');
            if (useDetectedOilCheckbox) {
                useDetectedOilCheckbox.disabled = false;
                document.getElementById('detectedOilInfo').textContent = 
                    `${result.oil_coordinates.length} oil points available for simulation`;
                document.getElementById('detectedOilInfo').style.display = 'block';
            }
        } else {
            detectionText.innerHTML = '<strong>MADOS Detection:</strong> No oil detected';
        }
        
    } catch (error) {
        console.error('Error with MADOS:', error);
        alert('Failed to process with MADOS: ' + error.message);
    } finally {
        hideLoadingOverlay();
    }
}

// Fungsi utilitas untuk menampilkan hasil optic di kolom Optical Processing Results
function tampilkanOpticResult(originalBase64, segmentationBase64) {
    const originalImage = document.getElementById('originalOpticImage');
    if (originalImage && originalBase64) {
        originalImage.src = originalBase64.startsWith('data:image/png;base64,') ? originalBase64 : `data:image/png;base64,${originalBase64}`;
        originalImage.style.display = 'block';
    }
    const segmentationImage = document.getElementById('madosSegmentationImage');
    if (segmentationImage && segmentationBase64) {
        segmentationImage.src = segmentationBase64.startsWith('data:image/png;base64,') ? segmentationBase64 : `data:image/png;base64,${segmentationBase64}`;
        segmentationImage.style.display = 'block';
    }
}

// Fungsi untuk mengatur opacity layer
function updateOpacity(layerType, value) {
    const layer = window.layers[layerType];
    const opacityControl = document.getElementById(layerType + 'Opacity');
    if (layer) {
        layer.setOpacity(value / 100);
        if (opacityControl) opacityControl.disabled = false;
    } else {
        if (opacityControl) opacityControl.disabled = true;
    }
}

// Setiap kali layer berhasil dibuat, aktifkan slider opacity
function enableOpacityControl(layerType, defaultValue) {
    const opacityControl = document.getElementById(layerType + 'Opacity');
    if (opacityControl) {
        opacityControl.disabled = false;
        opacityControl.value = defaultValue;
        updateOpacity(layerType, defaultValue);
    }
}

// Fungsi untuk toggle layer
function toggleLayer(layerType) {
    const button = document.getElementById(layerType + 'Toggle');
    if (!button) return;

    button.classList.toggle('active');
    const isActive = button.classList.contains('active');

    let layer = null;
    if (layerType === 'raster') {
        layer = window.layers.raster;
    } else if (layerType === 'segmentation') {
        layer = window.layers.segmentation;
    } else if (layerType === 'optic') {
        layer = window.layers.optic;
    } else if (layerType === 'mados') {
        layer = window.layers.mados;
    }
    if (layer) {
        if (isActive) {
            window.map.addLayer(layer);
            enableOpacityControl(layerType, document.getElementById(layerType + 'Opacity')?.value || 100);
        } else {
            window.map.removeLayer(layer);
        }
    }
}

// Fungsi untuk toggle basemap (split view)
function toggleBasemap(mode) {
    const mapContainer = document.getElementById('map').parentElement;
    const map2Container = document.getElementById('map2');
    const splitToggle = document.getElementById('splitToggle');

    if (mode === 'split' && !isSplitView) {
        // Aktifkan split view
        mapContainer.classList.add('split-view');
        map2Container.style.display = 'block';
        
        // Inisialisasi map kedua jika belum ada
        if (!map2) {
            map2 = L.map('map2').setView(window.map.getCenter(), window.map.getZoom());
            
            // Tambahkan Google Satellite sebagai basemap default untuk map2
            L.tileLayer('http://{s}.google.com/vt/lyrs=s&x={x}&y={y}&z={z}', {
                maxZoom: 20,
                subdomains: ['mt0', 'mt1', 'mt2', 'mt3'],
                attribution: '© Google'
            }).addTo(map2);
            
            // Sinkronkan pergerakan kedua map
            window.map.on('move', function() {
                map2.setView(window.map.getCenter(), window.map.getZoom(), {
                    animate: false
                });
            });
            
            map2.on('move', function() {
                window.map.setView(map2.getCenter(), map2.getZoom(), {
                    animate: false
                });
            });
        }
        
        // Tambahkan layer yang sama ke map2
        if (referenceRaster) {
            const rasterCopy = L.imageOverlay(referenceRaster._url, referenceRaster._bounds, {
                opacity: 1.0,
                className: 'original-overlay',
                interactive: true
            }).addTo(map2);
        }
        
        if (segmentationLayer) {
            const segmentationCopy = L.imageOverlay(segmentationLayer._url, segmentationLayer._bounds, {
                opacity: 0.8,
                className: 'segmentation-overlay',
                interactive: true,
                zIndex: 1000
            }).addTo(map2);
        }
        
        splitToggle.textContent = 'Single View';
        splitToggle.classList.add('active');
        isSplitView = true;
        
    } else {
        // Kembali ke tampilan tunggal
        mapContainer.classList.remove('split-view');
        map2Container.style.display = 'none';
        
        splitToggle.textContent = 'Split View';
        splitToggle.classList.remove('active');
        isSplitView = false;
    }
    
    // Perbarui ukuran peta
    window.map.invalidateSize();
    if (map2) map2.invalidateSize();
}

// Loading overlay functions
function showLoadingOverlay(message) {
    document.getElementById('loadingOverlay').style.display = 'flex';
    document.getElementById('loadingMessage').textContent = message || 'Loading...';
    document.getElementById('progressContainer').style.display = 'none';
}

function showLoadingProgress(message, percent) {
    document.getElementById('loadingOverlay').style.display = 'flex';
    document.getElementById('loadingMessage').textContent = message || 'Processing...';
    document.getElementById('progressContainer').style.display = 'flex';
    document.getElementById('progressBar').style.width = `${percent}%`;
    document.getElementById('progressText').textContent = `${percent}%`;
}

function updateLoadingOverlay(message, percent) {
    document.getElementById('loadingMessage').textContent = message || 'Processing...';
    if (percent !== undefined) {
        document.getElementById('progressContainer').style.display = 'flex';
        document.getElementById('progressBar').style.width = `${percent}%`;
        document.getElementById('progressText').textContent = `${percent}%`;
    }
}

function hideLoadingOverlay() {
    document.getElementById('loadingOverlay').style.display = 'none';
}

// Poll for job status
async function pollJobStatus(url, interval = 2000) {
    try {
        while (true) {
            const response = await fetch(url);
            if (!response.ok) {
                throw new Error(`Failed to check job status: ${response.status}`);
            }
            
            const data = await response.json();
            
            // Update progress if available
            if (data.progress) {
                updateLoadingOverlay(data.message || `Processing (${data.stage || 'step'} of 5)...`, data.progress);
            }
            
            // --- PATCH: Accept both 'completed' and 'success' as finished ---
            if (data.status === 'completed' || data.status === 'success') {
                // If result is directly in the response, return it
                if (data.result) {
                    return data.result;
                }
                // If data already contains the expected fields, return it
                if (data.original_image || data.segmentation_image || data.oil_coordinates) {
                    return data;
                }
                // Otherwise fetch result from result URL
                const resultUrl = data.result_url;
                if (resultUrl) {
                    const resultResponse = await fetch(resultUrl);
                    if (resultResponse.ok) {
                        return await resultResponse.json();
                    }
                }
                // If no result URL, try to get result directly
                const jobId = url.split('/').pop();
                const resultResponse = await fetch(`/mados_job_result/${jobId}`);
                if (resultResponse.ok) {
                    return await resultResponse.json();
                }
                // If still no result, wait and try again
                await new Promise(resolve => setTimeout(resolve, interval));
                continue;
            } else if (data.status === 'error') {
                throw new Error(data.error || data.message || 'An error occurred during processing');
            }
            // Wait before polling again
            await new Promise(resolve => setTimeout(resolve, interval));
        }
    } catch (error) {
        console.error('Error polling job status:', error);
        throw error;
    }
}

// Function for MADOS processing with progressive feedback
async function processMadosProgressively(formData) {
    showLoadingOverlay('Uploading files to MADOS service...');
    
    try {
        // Add progressive flag to request
        const response = await fetch('/predict_mados?progressive=true', {
            method: 'POST',
            body: formData
        });
        
        // Get response as text first for debugging
        const responseText = await response.text();
        
        // Try to parse as JSON
        let result;
        try {
            result = JSON.parse(responseText);
        } catch (parseError) {
            console.error('Failed to parse response as JSON:', responseText);
            throw new Error(`Failed to parse response: ${responseText.substring(0, 100)}...`);
        }
        
        // Check for error status
        if (result.status === 'error') {
            throw new Error(result.message || 'Unknown MADOS service error');
        }
        
        // Check if job is being processed in background
        if (result.status === 'processing' && result.job_id) {
            // Start polling for job status
            updateLoadingOverlay('Files received, processing started...', 10);
            
            // Poll until job completes
            return await pollJobStatus(result.check_url);
        } else {
            // Direct response (no background processing)
            return result;
        }
    } catch (error) {
        console.error('Error in MADOS processing:', error);
        hideLoadingOverlay();
        alert(`Error processing with MADOS: ${error.message}`);
        throw error;
    }
}

// Fungsi untuk memproses oil drift
async function processOilDrift() {
    const oilDriftInput = document.getElementById('oilDriftInput');
    const startTime = document.getElementById('startTime').value;
    const duration = document.getElementById('duration').value;
    
    if (!oilDriftInput.files[0]) {
        alert('Please select a CSV file');
        return;
    }
    
    if (!startTime) {
        alert('Please select a start time');
        return;
    }
    
    // Tampilkan loading overlay
    showLoadingOverlay('Simulating oil drift...');
    
    const formData = new FormData();
    formData.append('file', oilDriftInput.files[0]);
    formData.append('start_time', startTime);
    formData.append('duration', duration);
    
    // Tambahkan koordinat jika tersedia
    const lat = document.getElementById('lat').value;
    const lon = document.getElementById('lon').value;
    if (lat && lon) {
        formData.append('latitude', lat);
        formData.append('longitude', lon);
    }
    
    try {
        // Kirim request ke endpoint oil drift
        const response = await fetch('http://localhost:8000/simulate_oil_drift', {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`Server responded with status ${response.status}: ${errorText}`);
        }
        
        const result = await response.json();
        console.log('Received oil drift data:', result);
        
        if (!result.success) {
            throw new Error('Oil drift simulation failed');
        }
        
        // Initialize oilDriftLayer if it doesn't exist
        if (!window.map) {
            throw new Error('Map is not initialized');
        }
        
        // Hapus layer oil drift yang sudah ada
        if (oilDriftLayer) {
            window.map.removeLayer(oilDriftLayer);
        }
        
        if (oilDriftPath) {
            window.map.removeLayer(oilDriftPath);
        }
        
        // Buat layer untuk oil drift
        oilDriftLayer = L.layerGroup().addTo(window.map);
        
        // Tambahkan variabel untuk menyimpan data GeoJSON daratan
        let landPolygons = null;
        
        // Muat data GeoJSON daratan
        try {
            const landResponse = await fetch('https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_10m_land.geojson');
            if (!landResponse.ok) {
                throw new Error(`HTTP error! status: ${landResponse.status}`);
            }
            landPolygons = await landResponse.json();
            
            // Verifikasi data
            if (!landPolygons.features || landPolygons.features.length === 0) {
                console.warn('Data daratan tidak memiliki fitur');
            } else {
                console.log(`Data daratan memiliki ${landPolygons.features.length} fitur`);
            }
        } catch (error) {
            console.error('Tidak dapat memuat data daratan:', error);
            landPolygons = null;
        }
        
        // Fungsi untuk memeriksa apakah titik berada di daratan
        function isPointOnLand(lat, lng) {
            // Pastikan landPolygons dan turf tersedia
            if (!landPolygons || typeof turf === 'undefined') {
                console.warn('Data daratan atau library turf tidak tersedia');
                return false;
            }
            
            try {
                // Buat titik dengan format yang benar (longitude, latitude)
                const point = turf.point([lng, lat]);
                
                // Periksa apakah titik berada di dalam poligon daratan
                for (const feature of landPolygons.features) {
                    if (turf.booleanPointInPolygon(point, feature)) {
                        console.log(`Titik (${lat}, ${lng}) berada di daratan`);
                        return true; // Titik berada di daratan
                    }
                }
                
                console.log(`Titik (${lat}, ${lng}) berada di lautan`);
                return false; // Titik berada di lautan
            } catch (error) {
                console.error('Error saat memeriksa titik:', error);
                return false; // Anggap titik berada di lautan jika terjadi error
            }
        }
        
        // Persiapkan data untuk animasi
        const driftPoints = [];
        if (result.drift_points && Array.isArray(result.drift_points)) {
            result.drift_points.forEach((point) => {
                // Periksa apakah titik berada di daratan
                const isOnLand = isPointOnLand(point.latitude, point.longitude);
                
                // Hanya tambahkan titik jika berada di lautan
                if (!isOnLand) {
                    driftPoints.push(point);
                }
            });
            // Log jumlah titik untuk debugging
            console.log(`Processed ${driftPoints.length} drift points for animation`);

            // Fit map bounds to include all drift points
            const pathPoints = driftPoints.map(p => [p.latitude, p.longitude]);
            if (pathPoints.length > 0) {
                window.map.fitBounds(L.latLngBounds(pathPoints));
            }
            
            // Pastikan container slider ditampilkan sebelum setup
            const sliderContainer = document.getElementById('timeSliderContainer');
            if (sliderContainer) {
                sliderContainer.style.display = 'block';
            }
            
            // Setup time slider untuk animasi
            if (driftPoints.length > 0) {
                setupTimeSlider(driftPoints);
            } else {
                console.error('No valid drift points available for animation');
                alert('No valid drift points available for animation');
            }
        } else {
            console.error('Invalid drift points data format');
            alert('Invalid drift points data format');
        }
        
    } catch (error) {
        console.error('Error:', error);
        alert('Failed to process oil drift: ' + error.message);
    } finally {
        // Hide loading overlay
        hideLoadingOverlay();
    }
}

// Simulasi OpenOil yang sudah diupdate untuk menggunakan gabungan koordinat
async function simulateOpenOil() {
    showLoadingOverlay('Simulating oil drift with OpenOil...');
    
    const useDetectedOil = document.getElementById('useDetectedOil').checked;
    const latitude = parseFloat(document.getElementById('lat').value);
    const longitude = parseFloat(document.getElementById('lon').value);
    const oilType = document.getElementById('oilType').value;
    const radius = parseInt(document.getElementById('radius').value || 3000);
    const numParticles = parseInt(document.getElementById('numParticles').value || 1000);
    const durationHours = parseInt(document.getElementById('durationHours').value || 24);
    const startTime = document.getElementById('sarStartTime').value;
    
    // Validate start time
    if (!startTime) {
        showError('Please enter a start time for the simulation');
        hideLoadingOverlay();
        return;
    }
    
    // Gabungkan koordinat dari SAR dan MADOS
    const combinedOilCoordinates = [...detectedOilCoordinates, ...madosOilCoordinates];
    
    if (!useDetectedOil && (isNaN(latitude) || isNaN(longitude))) {
        alert('Please enter valid latitude and longitude or use detected oil');
        hideLoadingOverlay();
        return;
    }
    
    try {
        const formData = new FormData();
        formData.append('oil_type', oilType);
        formData.append('radius', radius);
        formData.append('num_particles', numParticles);
        formData.append('duration_hours', durationHours);
        if (startTime) {
            formData.append('start_time', startTime);
        }

        // Add the boolean flag for using detected oil
        formData.append('use_detected_oil', useDetectedOil);

        // Handle coordinates based on checkbox state
        if (useDetectedOil && combinedOilCoordinates.length > 0) {
            console.log('Sending combined oil coordinates:', combinedOilCoordinates);
            formData.append('oil_coordinates', JSON.stringify(combinedOilCoordinates));
        } else if (!useDetectedOil && !isNaN(latitude) && !isNaN(longitude)) {
             console.log('Sending manual coordinates:', { latitude: latitude, longitude: longitude });
            formData.append('latitude', latitude);
            formData.append('longitude', longitude);
        } else {
             throw new Error('No valid coordinates provided or detected.');
        }
        
        const response = await fetch('http://localhost:8000/openoil_simulation', {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`Server responded with status ${response.status}: ${errorText}`);
        }
        
        const result = await response.json();
        
        if (!result.success) {
            throw new Error('OpenOil simulation failed');
        }
        
        // Remove existing oil drift layer
        if (oilDriftLayer) {
            window.map.removeLayer(oilDriftLayer);
        }
        
        oilDriftLayer = L.layerGroup().addTo(window.map);
        
        // Add trajectory visualization
        const pathPoints = [];
        result.trajectory.forEach((timePoint) => {
            timePoint.points.forEach((point) => {
                const marker = L.circleMarker([point.latitude, point.longitude], {
                    radius: 3,
                    fillColor: '#00FFFF',
                    color: '#000',
                    weight: 1,
                    opacity: 1,
                    fillOpacity: 0.8
                }).addTo(oilDriftLayer);
                
                marker.bindPopup(`
                    <strong>Time:</strong> ${timePoint.time}<br>
                    <strong>Position:</strong> ${point.latitude.toFixed(6)}, ${point.longitude.toFixed(6)}<br>
                    <strong>Depth:</strong> ${point.depth.toFixed(2)} m
                `);
                
                pathPoints.push([point.latitude, point.longitude]);
            });
        });
        
        if (pathPoints.length > 0) {
            window.map.fitBounds(L.latLngBounds(pathPoints));
        }
        
        // Display results
        displayOpenOilResult(result);
        
        // Show info about the simulation (SAR)
        if (result.metadata) {
            safeSetInnerHTML('sarOpenOilText', `
                <strong>Simulation Details:</strong><br>
                - Source: ${result.metadata.source || 'SAR'}<br>
                - Oil Type: ${result.metadata.oil_type}<br>
                - Number of Particles: ${result.metadata.num_particles}<br>
                - Duration: ${result.metadata.duration_hours} hours<br>
                - Number of Oil Points: ${result.metadata.num_oil_points || '-'}`
        );
        safeSetDisplay('sarOpenOilInfo', 'block');
    }
        
    } catch (error) {
        console.error('Error:', error);
        alert('Failed to simulate with OpenOil: ' + error.message);
    } finally {
        hideLoadingOverlay();
    }
}

// Function to load oil types for OpenOil
async function loadOilTypes() {
    try {
        const response = await fetch('http://localhost:8000/available_oil_types');
        
        if (!response.ok) {
            throw new Error(`Server responded with status ${response.status}`);
        }
        
        const result = await response.json();
        
        if (!result.success) {
            throw new Error('Failed to load oil types');
        }
        
        const oilTypeSelect = document.getElementById('oilType');
        oilTypeSelect.innerHTML = '';
        
        result.oil_types.forEach((oilType) => {
            const option = document.createElement('option');
            option.value = oilType;
            option.textContent = oilType;
            oilTypeSelect.appendChild(option);
        });
        
    } catch (error) {
        console.error('Error loading oil types:', error);
    }
}

// Add this near the top of the file with other global variables
let oilBoomLayersMap = {};

async function addOilBoomLayers() {
    console.log("Adding Oil Boom Layers...");
    const oilBoomLayers = {
        Gudang_Oil_Boom: {
            url: '/static/data/Oil-GeoJSON/Gudang Oil Boom (PT).geojson',
            style: { color: '#ff0000', weight: 2, opacity: 0.8 },
            icon: '/static/images/Gudang_Oil_Boom.png' // Custom icon for Gudang
        },
        Oil_Boom: {
            url: '/static/data/Oil-GeoJSON/Oil Boom Latest.geojson',
            style: { color: '#00ff00', weight: 2, opacity: 0.8 }
        },
        Skimming_Boat: {
            url: '/static/data/Oil-GeoJSON/Skimming Boat Latest.geojson',
            style: { color: '#0000ff', weight: 2, opacity: 0.8 },
            icon: '/static/images/ferry.png' // Custom icon for Skimming Boat
        }
    };

    const customIcon = (iconUrl) => {
        return L.icon({
            iconUrl: iconUrl,
            iconSize: [25, 25], // size of the icon
            iconAnchor: [12, 25], // point of the icon which will correspond to marker's location
            popupAnchor: [0, -20] // point from which the popup should open relative to the iconAnchor
        });
    };

    for (const key in oilBoomLayers) {
        const layerInfo = oilBoomLayers[key];
        try {
            const response = await fetch(layerInfo.url);
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            const data = await response.json();
            console.log(`Successfully loaded ${key} data:`, data);

            oilBoomLayersMap[key] = L.geoJSON(data, {
                style: function(feature) {
                    return layerInfo.style;
                },
                // Use custom icon if defined, otherwise use default marker
                pointToLayer: function(feature, latlng) {
                    if (layerInfo.icon) {
                        return L.marker(latlng, { icon: customIcon(layerInfo.icon) });
                    }
                    return L.marker(latlng); // Default marker
                },
                onEachFeature: function(feature, layer) {
                    // Add popup with properties if available
                    if (feature.properties) {
                        let popupContent = `<b>${key}</b><br>`;
                        for (const prop in feature.properties) {
                            popupContent += `<b>${prop}:</b> ${feature.properties[prop]}<br>`;
                        }
                        layer.bindPopup(popupContent);
                    }
                }
            });

            // Add layer to map initially (you might want to change this based on default visibility)
            oilBoomLayersMap[key].addTo(map);
            console.log(`Layer ${key} added to map`);

        } catch (error) {
            console.error(`Error loading or adding layer ${key}:`, error);
            showError(`Failed to load layer "${key}". Please check data file and console for details.`);
        }
    }
}

// Fungsi untuk toggle visibilitas layer Oil Boom
function toggleOilBoomLayer(key) {
    const layer = oilBoomLayersMap[key];
    // Use the correct button ID based on the HTML
    const buttonId = key === 'Gudang_Oil_Boom' ? 'gudangBoomToggle' :
                     key === 'Oil_Boom' ? 'oilBoomToggle' :
                     key === 'Skimming_Boat' ? 'skimmingBoatToggle' : null;

    if (!buttonId) {
        console.error(`Unknown oil boom layer key: ${key}`);
        return;
    }

    const button = document.getElementById(buttonId);

    if (layer) {
        if (map.hasLayer(layer)) {
            map.removeLayer(layer);
            if (button) button.classList.remove('active');
            console.log(`Layer ${key} removed from map.`);
        } else {
            layer.addTo(map);
            if (button) button.classList.add('active');
            console.log(`Layer ${key} added to map.`);
        }
    } else {
        console.warn(`Layer ${key} not found in oilBoomLayersMap.`);
    }
}

// Fungsi untuk mengatur opacity layer Oil Boom
function updateOilBoomOpacity(key, value) {
    const layer = oilBoomLayersMap[key];
    if (layer) {
        layer.setStyle({
            opacity: value / 100,
            fillOpacity: value / 100
        });
    }
}

// Function to load and display AOI layer
function loadAOILayer() {
    fetch('/static/data/geojson/AOI.geojson')
        .then(response => {
            if (!response.ok) throw new Error('AOI GeoJSON not found');
            return response.json();
        })
        .then(data => {
            const aoiLayer = L.geoJSON(data, {
                style: {
                    color: '#000000',
                    weight: 2,
                    opacity: 0.8,
                    fillOpacity: 0
                },
                onEachFeature: function(feature, layer) {
                    if (feature.properties) {
                        let popupContent = '';
                        for (const key in feature.properties) {
                            popupContent += `<strong>${key}:</strong> ${feature.properties[key]}<br>`;
                        }
                        if (popupContent) layer.bindPopup(popupContent);
                    }
                }
            }).addTo(window.map);

            // Add to layer control if it exists
            if (window.layerControl) {
                window.layerControl.addOverlay(aoiLayer, 'Area of Interest');
            }

            // Fit map to AOI bounds
            window.map.fitBounds(aoiLayer.getBounds());
            enableOpacityControl('mados', 80);
        })
        .catch(err => console.warn('AOI GeoJSON error:', err));
}

// Fungsi untuk menambahkan slider waktu
function addTimeSlider(trajectoryData) {
    console.log('Adding time slider with trajectory data:', trajectoryData);
    if (trajectoryData.length > 0) {
        console.log('First time point:', trajectoryData[0]);
        console.log('Sample point:', trajectoryData[0].points ? trajectoryData[0].points[0] : 'No points');
    }
    // Hapus slider yang sudah ada jika ada
    const existingSlider = document.getElementById('timeSliderContainer');
    if (existingSlider) {
        existingSlider.remove();
    }
    
    // Buat container untuk slider
    const sliderContainer = document.createElement('div');
    sliderContainer.id = 'timeSliderContainer';
    sliderContainer.className = 'time-slider-container';
    
    // Tambahkan judul
    const title = document.createElement('h4');
    title.textContent = 'Oil Spill Movement Animation';
    title.style.margin = '0 0 10px 0';
    sliderContainer.appendChild(title);
    
    // Tambahkan display waktu
    const timeDisplay = document.createElement('div');
    timeDisplay.id = 'timeDisplay';
    timeDisplay.className = 'time-display';
    sliderContainer.appendChild(timeDisplay);
    
    // Tambahkan slider
    const slider = document.createElement('input');
    slider.type = 'range';
    slider.min = '0';
    slider.max = (trajectoryData.length - 1).toString();
    slider.value = '0';
    slider.className = 'time-slider';
    slider.id = 'timeSlider';
    sliderContainer.appendChild(slider);
    
    // Tambahkan kontrol animasi
    const controls = document.createElement('div');
    controls.className = 'time-controls';
    
    const playButton = document.createElement('button');
    playButton.textContent = 'Play';
    playButton.id = 'playBtn'; // Changed to match the ID used elsewhere
    
    const pauseButton = document.createElement('button');
    pauseButton.textContent = 'Pause';
    pauseButton.id = 'pauseBtn'; // Changed to match the ID used elsewhere
    pauseButton.disabled = true;
    
    const resetButton = document.createElement('button');
    resetButton.textContent = 'Reset';
    resetButton.id = 'resetBtn'; // Changed to match the ID used elsewhere
    
    controls.appendChild(playButton);
    controls.appendChild(pauseButton);
    controls.appendChild(resetButton);
    sliderContainer.appendChild(controls);
    
    // Tambahkan container ke peta
    document.querySelector('.map-container').appendChild(sliderContainer);
    
    // Tampilkan waktu awal
    updateTimeDisplay(trajectoryData[0].time);
    
    // Tampilkan titik-titik awal
    updateMapPoints(trajectoryData[0]);
    
    // Event listener untuk slider
    slider.addEventListener('input', function() {
        console.log('Slider moved to:', this.value);
        currentTimeIndex = parseInt(this.value);
        console.log('Current time index:', currentTimeIndex);
        console.log('Trajectory data length:', trajectoryData.length);
        console.log('Time point data:', trajectoryData[currentTimeIndex]);
        
        // Pastikan nilai valid
        if (currentTimeIndex >= 0 && currentTimeIndex < trajectoryData.length) {
            updateMapPoints(trajectoryData[currentTimeIndex]);
            updateTimeDisplay(trajectoryData[currentTimeIndex].time);
        } else {
            console.error('Invalid time index:', currentTimeIndex);
        }
    });
    
    // Event listener untuk tombol play
    playButton.addEventListener('click', function() {
        this.disabled = true;
        pauseButton.disabled = false;
        
        // Mulai animasi
        timeSliderInterval = setInterval(function() {
            currentTimeIndex++;
            if (currentTimeIndex >= trajectoryData.length) {
                currentTimeIndex = 0;
            }
            
            slider.value = currentTimeIndex;
            updateMapPoints(trajectoryData[currentTimeIndex]);
            updateTimeDisplay(trajectoryData[currentTimeIndex].time);
        }, 1000); // Update animasi setiap 1 detik, tetapi data tetap interval 30 menit
    });
    
    // Event listener untuk tombol pause
    pauseButton.addEventListener('click', function() {
        this.disabled = true;
        playButton.disabled = false;
        
        // Hentikan animasi
        clearInterval(timeSliderInterval);
    });
    
    // Event listener untuk tombol reset
    resetButton.addEventListener('click', function() {
        // Hentikan animasi jika sedang berjalan
        clearInterval(timeSliderInterval);
        
        // Reset ke awal
        currentTimeIndex = 0;
        slider.value = currentTimeIndex;
        updateMapPoints(trajectoryData[currentTimeIndex]);
        updateTimeDisplay(trajectoryData[currentTimeIndex].time);
        
        // Reset tombol
        playButton.disabled = false;
        pauseButton.disabled = true;
    });
}

// Fungsi untuk memperbarui tampilan waktu
function updateTimeDisplay(timeString) {
    const timeDisplay = document.getElementById('timeDisplay');
    if (timeDisplay) {
        // Use the exact time string from the backend
        timeDisplay.textContent = `Time: ${timeString}`;
        
        // Optional: You could parse and reformat if a specific locale is needed, 
        // but using the provided string ensures accuracy with backend steps.
        // try {
        //     const date = new Date(timeString);
        //     if (!isNaN(date.getTime())) {
        //         timeDisplay.textContent = `Time: ${date.toLocaleString()}`; // Use locale-specific formatting
        //     } else {
        //         timeDisplay.textContent = `Time: Invalid Date`;
        //     }
        // } catch (e) {
        //      timeDisplay.textContent = `Time: Error formatting date`;
        //      console.error('Error formatting date:', e);
        // }
    }
}

// Fungsi untuk memperbarui titik-titik pada peta
function updateMapPoints(timePoint) {
    console.log('Updating map points for time:', timePoint.time);
    
    try {
        // Validasi data timePoint
        if (!timePoint) {
            console.error('Error: timePoint is undefined or null');
            return;
        }
        
        console.log('Number of points:', timePoint.points ? timePoint.points.length : 'No points');
        
        // Hapus layer yang sudah ada
        if (oilDriftLayer) {
            oilDriftLayer.clearLayers(); // Gunakan clearLayers() daripada removeLayer
        } else {
            // Buat layer baru jika belum ada
            oilDriftLayer = L.layerGroup().addTo(window.map);
        }
        
        // Tambahkan titik-titik untuk waktu ini
        if (timePoint.points && timePoint.points.length > 0) {
            timePoint.points.forEach(point => {
                // Tambahkan marker
                const marker = L.circleMarker([point.latitude, point.longitude], {
                    radius: 3,
                    fillColor: '#00FFFF',
                    color: '#000',
                    weight: 1,
                    opacity: 1,
                    fillOpacity: 0.8
                }).addTo(oilDriftLayer);
                
                // Tambahkan popup dengan tag penutup yang benar
                marker.bindPopup(`
                    <strong>Time:</strong> ${timePoint.time}<br>
                    <strong>Position:</strong> ${point.latitude.toFixed(6)}, ${point.longitude.toFixed(6)}<br>
                    <strong>Depth:</strong> ${point.depth ? point.depth.toFixed(2) : '0.00'} m
                `);
            });
            console.log(`Added ${timePoint.points.length} points to map`);
        } else {
            console.warn('No points available for this time point');
        }
    } catch (error) {
        console.error('Error updating map points:', error);
    }
}

let oilDriftAnimation = {
    timer: null,
    currentIndex: 0,
    playing: false,
    driftPoints: [],
    markers: [],
    path: null
};

function setupTimeSlider(driftPoints) {
    // Pastikan driftPoints valid dan memiliki data
    if (!driftPoints || driftPoints.length === 0) {
        console.error('No drift points data available for time slider');
        return;
    }

    console.log('Setting up time slider with', driftPoints.length, 'points');
    
    const sliderContainer = document.getElementById('timeSliderContainer');
    if (!sliderContainer) {
        console.error('Time slider container not found');
        return;
    }
    
    const slider = document.getElementById('timeSlider');
    const timeDisplay = document.getElementById('timeDisplay');
    
    // Coba kedua kemungkinan ID untuk tombol kontrol
    const playBtn = document.getElementById('playBtn') || document.getElementById('playButton');
    const pauseBtn = document.getElementById('pauseBtn') || document.getElementById('pauseButton');
    const resetBtn = document.getElementById('resetBtn') || document.getElementById('resetButton');

    // Verifikasi semua elemen UI ditemukan
    if (!slider || !timeDisplay || !playBtn || !pauseBtn || !resetBtn) {
        console.error('Some time slider elements not found:', {
            slider: !!slider,
            timeDisplay: !!timeDisplay,
            playBtn: !!playBtn,
            pauseBtn: !!pauseBtn,
            resetBtn: !!resetBtn
        });
        return;
    }

    // Pastikan container ditampilkan
    sliderContainer.style.display = 'block';

    // --- RESAMPLE DATA TO 30-MINUTE INTERVALS ---
    // 1. Ambil waktu awal dan akhir
    const parseTime = t => (t instanceof Date ? t : new Date(t));
    const startTime = parseTime(driftPoints[0].time);
    const endTime = parseTime(driftPoints[driftPoints.length - 1].time);

    // 2. Buat array waktu per 30 menit
    const resampledPoints = [];
    let t = new Date(startTime);
    while (t <= endTime) {
        // 3. Cari dua titik terdekat sebelum dan sesudah t
        let before = null, after = null;
        for (let i = 0; i < driftPoints.length; i++) {
            const ptTime = parseTime(driftPoints[i].time);
            if (ptTime <= t) before = driftPoints[i];
            if (ptTime >= t) {
                after = driftPoints[i];
                break;
            }
        }
        if (!before) before = driftPoints[0];
        if (!after) after = driftPoints[driftPoints.length - 1];

        // 4. Interpolasi linier jika perlu
        let interp = {};
        if (before === after) {
            interp = {...before};
        } else {
            const t0 = parseTime(before.time).getTime();
            const t1 = parseTime(after.time).getTime();
            const ratio = (t.getTime() - t0) / (t1 - t0);
            interp = {
                time: t.toISOString(),
                latitude: before.latitude + (after.latitude - before.latitude) * ratio,
                longitude: before.longitude + (after.longitude - before.longitude) * ratio,
                u_current: before.u_current + (after.u_current - before.u_current) * ratio,
                v_current: before.v_current + (after.v_current - before.v_current) * ratio
            };
        }
        resampledPoints.push(interp);
        // Tambah 30 menit
        t = new Date(t.getTime() + 30 * 60 * 1000);
    }

    oilDriftAnimation.driftPoints = resampledPoints;
    oilDriftAnimation.currentIndex = 0;
    oilDriftAnimation.playing = false;

    slider.max = resampledPoints.length - 1;
    slider.value = 0;
    updateOilDriftFrame(0);

    slider.oninput = function() {
        updateOilDriftFrame(Number(slider.value));
    };

    playBtn.onclick = function() {
        if (!oilDriftAnimation.playing) {
            oilDriftAnimation.playing = true;
            playBtn.disabled = true;
            pauseBtn.disabled = false;
            animateOilDrift();
        }
    };
    
    pauseBtn.onclick = function() {
        oilDriftAnimation.playing = false;
        playBtn.disabled = false;
        pauseBtn.disabled = true;
        if (oilDriftAnimation.timer) clearTimeout(oilDriftAnimation.timer);
    };
    
    resetBtn.onclick = function() {
        oilDriftAnimation.playing = false;
        playBtn.disabled = false;
        pauseBtn.disabled = true;
        slider.value = 0;
        updateOilDriftFrame(0);
        if (oilDriftAnimation.timer) clearTimeout(oilDriftAnimation.timer);
    };
}

function updateOilDriftFrame(index) {
    const driftPoints = oilDriftAnimation.driftPoints;
    const timeDisplay = document.getElementById('timeDisplay');
    const slider = document.getElementById('timeSlider');
    
    // Remove old markers
    if (oilDriftAnimation.markers) {
        oilDriftAnimation.markers.forEach(m => window.map.removeLayer(m));
    }
    oilDriftAnimation.markers = [];
    
    // Remove old path (tetap hapus path lama jika ada)
    if (oilDriftAnimation.path) {
        window.map.removeLayer(oilDriftAnimation.path);
        oilDriftAnimation.path = null;
    }
    
    // Add new markers for current frame
    if (driftPoints.length > 0 && index < driftPoints.length) {
        // Tampilkan titik oil spill saat ini
        const currentPoint = driftPoints[index];
        
        // Buat marker untuk titik saat ini dengan ukuran yang lebih besar
        const marker = L.circleMarker([currentPoint.latitude, currentPoint.longitude], {
            radius: 12, // Ukuran lebih besar untuk visibilitas yang lebih baik
            fillColor: '#00FFFF',
            color: '#000',
            weight: 1,
            opacity: 1,
            fillOpacity: 0.8,
            className: 'oil-drift-marker'
        }).addTo(window.map);
        
        // Tambahkan popup dengan informasi
        marker.bindPopup(`
            <strong>Time:</strong> ${currentPoint.time}<br>
            <strong>Position:</strong> ${currentPoint.latitude.toFixed(6)}, ${currentPoint.longitude.toFixed(6)}<br>
            <strong>Current:</strong> u=${currentPoint.u_current.toFixed(3)}, v=${currentPoint.v_current.toFixed(3)}
        `);
        
        oilDriftAnimation.markers.push(marker);
        
        // Tambahkan panah vektor untuk menunjukkan arah arus
        if (currentPoint.u_current !== 0 || currentPoint.v_current !== 0) {
            // Hitung panjang panah berdasarkan kekuatan arus
            const scaleFactor = 0.12;
            
            // Hitung koordinat ujung panah
            const endLat = currentPoint.latitude + (currentPoint.v_current * scaleFactor);
            const endLng = currentPoint.longitude + (currentPoint.u_current * scaleFactor);
            
            // Buat panah menggunakan Leaflet.polyline
            const arrowLine = L.polyline([
                [currentPoint.latitude, currentPoint.longitude],
                [endLat, endLng]
            ], {
                color: '#FF0000',
                weight: 3,
                opacity: 1.0
            }).addTo(window.map);
            
            oilDriftAnimation.markers.push(arrowLine);
            
            // Tambahkan arrowhead
            const arrowHead = L.circleMarker([endLat, endLng], {
                radius: 2,
                fillColor: '#FF0000',
                color: '#000000',
                weight: 1,
                opacity: 1,
                fillOpacity: 1
            }).addTo(window.map);
            
            oilDriftAnimation.markers.push(arrowHead);
        }
        
        timeDisplay.textContent = `Time: ${currentPoint.time}`;
        slider.value = index;
    } else {
        timeDisplay.textContent = 'Time: -';
    }
}

function animateOilDrift() {
    if (!oilDriftAnimation.playing) return;
    
    let idx = oilDriftAnimation.currentIndex;
    if (idx < oilDriftAnimation.driftPoints.length - 1) {
        idx++;
        oilDriftAnimation.currentIndex = idx;
        updateOilDriftFrame(idx);
        
        // Atur interval waktu untuk animasi
        // Tampilkan setiap frame selama 2 detik untuk memberikan waktu yang cukup untuk melihat pergerakan
        oilDriftAnimation.timer = setTimeout(animateOilDrift, 2000);
    } else {
        oilDriftAnimation.playing = false;
        document.getElementById('playBtn').disabled = false;
        document.getElementById('pauseBtn').disabled = true;
    }
}

// Add support for progressive processing of MADOS results
async function uploadMadosWithProgress(formData) {
    showLoadingOverlay('Uploading files to MADOS service...');
    
    try {
        // Add progressive flag to enable background processing
        const response = await fetch('/predict_mados?progressive=true', {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            const errorData = await response.json();
            
            // Check if service was restarted
            if (errorData.service_restarted) {
                // Wait for service to initialize and try again
                showLoadingOverlay('MADOS service was restarted. Waiting for initialization...', 30);
                await new Promise(resolve => setTimeout(resolve, 15000)); // Wait 15 seconds
                
                // Try one more time
                showLoadingOverlay('Retrying upload after service restart...', 40);
                const retryResponse = await fetch('/predict_mados?progressive=true', {
                    method: 'POST',
                    body: formData
                });
                
                if (!retryResponse.ok) {
                    const retryErrorData = await retryResponse.json();
                    throw new Error(retryErrorData.detail || 'Upload failed after service restart');
                }
                
                const retryData = await retryResponse.json();
                
                if (retryData.status === 'processing' && retryData.job_id) {
                    // Start polling for job status
                    updateLoadingOverlay('Files uploaded, processing started...', 10);
                    
                    // Clean memory on service before starting heavy processing
                    fetch('/clean_memory').catch(err => console.warn('Failed to clean memory:', err));
                    
                    return pollMadosJobStatus(retryData.job_id);
                } else {
                    // Direct result
                    return retryData;
                }
            }
            
            throw new Error(errorData.detail || 'Upload failed');
        }
        
        const data = await response.json();
        
        if (data.status === 'processing' && data.job_id) {
            // Start polling for job status
            updateLoadingOverlay('Files uploaded, processing started...', 10);
            
            // Clean memory on service before starting heavy processing
            fetch('/clean_memory').catch(err => console.warn('Failed to clean memory:', err));
            
            return pollMadosJobStatus(data.job_id);
        } else {
            // Direct result
            return data;
        }
    } catch (error) {
        hideLoadingOverlay();
        showError(`Error uploading to MADOS: ${error.message}`);
        throw error;
    }
}

// Poll for job status until complete
async function pollMadosJobStatus(jobId) {
    let retryCount = 0;
    let connectionFailures = 0;
    const maxRetries = 180; // 15 minutes (5s * 180)
    const maxConnectionFailures = 5; // Allow up to 5 consecutive connection failures
    const pollingInterval = 5000; // 5 seconds
    let lastSuccessTime = Date.now();
    
    while (retryCount < maxRetries) {
        try {
            const response = await fetch(`/mados_job_status/${jobId}`);
            
            // Reset connection failures counter on successful request
            connectionFailures = 0;
            lastSuccessTime = Date.now();
            
            if (!response.ok) throw new Error('Failed to check job status');
            
            const status = await response.json();
            
            // Update loading message with progress indicator
            if (status.progress !== undefined) {
                updateLoadingOverlay(status.message || 'Processing...', status.progress);
            } else {
                updateLoadingOverlay(status.message || 'Processing...');
            }
            
            // Check if job is complete
            if (status.status === 'completed') {
                // Fetch final result
                try {
                    const resultResponse = await fetch(`/mados_job_result/${jobId}`);
                    if (!resultResponse.ok) throw new Error('Failed to fetch results');
                    return await resultResponse.json();
                } catch (resultError) {
                    console.error('Error fetching results:', resultError);
                    
                    // Try again after a short delay
                    await new Promise(resolve => setTimeout(resolve, 2000));
                    const retryResultResponse = await fetch(`/mados_job_result/${jobId}`);
                    if (!retryResultResponse.ok) throw new Error('Failed to fetch results on retry');
                    return await retryResultResponse.json();
                }
            }
            
            // Check if job failed
            if (status.status === 'error') {
                throw new Error(status.error || 'Processing failed');
            }
            
            // Check if job is in recovery mode
            if (status.status === 'recovering' || status.status === 'restarting') {
                updateLoadingOverlay(`${status.message || 'Recovering from an issue...'}`, status.progress || 50);
            }
            
            // Wait before polling again
            await new Promise(resolve => setTimeout(resolve, pollingInterval));
            retryCount++;
        } catch (error) {
            // Count connection failures
            if (error.message.includes('Failed to fetch') || 
                error.message.includes('NetworkError') || 
                error.message.includes('Failed to check job status')) {
                connectionFailures++;
                const timeSinceLastSuccess = Math.floor((Date.now() - lastSuccessTime) / 1000);
                
                console.warn(`Connection failure ${connectionFailures}/${maxConnectionFailures} (${timeSinceLastSuccess}s since last success)`);
                
                if (connectionFailures >= maxConnectionFailures) {
                    // Try to restart the service via the endpoint
                    updateLoadingOverlay('Connection to MADOS service lost. Attempting to restart...', 30);
                    
                    try {
                        // Make a request to clean_memory which will attempt to restart the service if it's down
                        const restartResponse = await fetch('/clean_memory');
                        
                        // Wait for service to restart (longer time)
                        await new Promise(resolve => setTimeout(resolve, 30000)); // 30 seconds
                        
                        // After restart, try to check if the job is still there
                        try {
                            const checkResponse = await fetch(`/mados_job_status/${jobId}`);
                            if (checkResponse.ok) {
                                connectionFailures = 0;
                                updateLoadingOverlay('MADOS service restarted. Continuing to check job status...', 40);
                                continue;
                            }
                        } catch (checkError) {
                            console.warn('Job status check after restart failed:', checkError);
                        }
                        
                        // If we've lost the job, we need to restart from scratch
                        updateLoadingOverlay('Lost connection to job. Preparing to restart processing...', 20);
                        return { status: 'restart_needed', message: 'Connection to MADOS service lost. Please try again.' };
                    } catch (restartError) {
                        console.error('Error restarting service:', restartError);
                    }
                }
            }
            
            if (connectionFailures >= maxConnectionFailures) {
                hideLoadingOverlay();
                showError(`MADOS service appears to be down. Please restart the application and try again.`);
                throw new Error('MADOS service connection lost');
            }
            
            console.warn(`Error checking job status (attempt ${retryCount + 1}): ${error.message}`);
            // Continue retrying without showing error to user
            await new Promise(resolve => setTimeout(resolve, pollingInterval));
            retryCount++;
        }
    }
    
    hideLoadingOverlay();
    throw new Error('Job processing timed out after 15 minutes');
}

// Modify detectOilSpillMados to use progressive processing
async function detectOilSpillMados() {
    try {
        const madosFilesInput = document.getElementById('madosFiles');
        
        if (!madosFilesInput.files || madosFilesInput.files.length === 0) {
            showError('Please select files for MADOS detection');
            return;
        }
        
        const formData = new FormData();
        
        // Check if it's a single ZIP file or multiple TIF files
        if (madosFilesInput.files.length === 1 && 
            madosFilesInput.files[0].name.toLowerCase().endsWith('.zip')) {
            // Single ZIP file
            formData.append('file', madosFilesInput.files[0]);
        } else {
            // Multiple files
            for (let i = 0; i < madosFilesInput.files.length; i++) {
                formData.append('files', madosFilesInput.files[i]);
            }
        }
        
        // Use progressive processing to reduce UI lag
        let maxRetries = 3;
        let retryCount = 0;
        
        while (retryCount < maxRetries) {
            try {
                const result = await uploadMadosWithProgress(formData);
                
                // Check if we need to restart from scratch due to service restart
                if (result.status === 'restart_needed') {
                    console.warn('Need to restart processing from scratch');
                    retryCount++;
                    
                    if (retryCount >= maxRetries) {
                        throw new Error('Too many restart attempts. Please try again later.');
                    }
                    
                    updateLoadingOverlay(`Retrying processing (attempt ${retryCount}/${maxRetries})...`, 10);
                    await new Promise(resolve => setTimeout(resolve, 5000)); // Wait 5 seconds before retry
                    continue;
                }
                
                // Process result same as before
                if (result.status === 'success') {
                    // Display the segmentation image
                    document.getElementById('resultMados').src = result.segmentation_image;
                    document.getElementById('resultMados').style.display = 'block';
                    document.getElementById('resultMadosContainer').style.display = 'block';
                    
                    // Show original image if available
                    if (result.original_image) {
                        document.getElementById('originalMados').src = result.original_image;
                        document.getElementById('originalMados').style.display = 'block';
                        document.getElementById('originalMadosContainer').style.display = 'block';
                    }
                    
                    // Store the oil coordinates for simulation
                    window.madosOilCoordinates = result.oil_coordinates || [];
                    
                    // Update UI with oil pixel count
                    const oilPixelCount = document.getElementById('madosOilPixelCount');
                    if (oilPixelCount) {
                        oilPixelCount.textContent = `Oil pixels detected: ${result.num_oil_pixels || 0}`;
                        oilPixelCount.style.display = 'block';
                    }
                    
                    // Enable OpenOil simulation with MADOS results
                    const useMADOSCheckbox = document.getElementById('useMADOS');
                    if (useMADOSCheckbox) {
                        useMADOSCheckbox.disabled = false;
                        useMADOSCheckbox.checked = true;  // Auto-enable using MADOS results
                    }
                    
                    // Clean up memory on server
                    fetch('/clean_memory').catch(err => console.warn('Failed to clean memory:', err));
                    
                    // Show success message
                    showSuccess('MADOS oil detection completed successfully!');
                    break; // Exit the retry loop on success
                } else {
                    showError(`MADOS detection failed: ${result.message || 'Unknown error'}`);
                    break; // Exit the retry loop on definitive error
                }
            } catch (error) {
                console.error(`MADOS processing error (attempt ${retryCount + 1}/${maxRetries}):`, error);
                retryCount++;
                
                if (retryCount >= maxRetries) {
                    hideLoadingOverlay();
                    showError(`Error in MADOS detection after ${maxRetries} attempts: ${error.message}`);
                    throw error;
                }
                
                // Wait before retrying
                updateLoadingOverlay(`Processing error, retrying in 5 seconds (attempt ${retryCount}/${maxRetries})...`, 10);
                await new Promise(resolve => setTimeout(resolve, 5000));
            }
        }
        
        hideLoadingOverlay();
    } catch (error) {
        hideLoadingOverlay();
        showError(`Error in MADOS detection: ${error.message}`);
        console.error('MADOS detection error:', error);
    }
}

// Full sequential upload workflow for MADOS
async function processOpticSequentialFull() {
    const input = document.getElementById('opticInput');
    if (!input.files || input.files.length < 2) {
        alert('Please select at least 2 band files for sequential processing');
        return;
    }

    showLoadingOverlay('Starting sequential upload...');

    try {
        // Sort files by band order
        const bandOrder = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B11', 'B12'];
        const sortedFiles = Array.from(input.files).sort((a, b) => {
            const aBand = bandOrder.findIndex(band => a.name.includes(band));
            const bBand = bandOrder.findIndex(band => b.name.includes(band));
            return aBand - bBand;
        });

        // 1. Start with the first band
        let formData = new FormData();
        formData.append('file', sortedFiles[0]);
        formData.append('sequential_mode', 'start');

        let response = await fetch('/predict_mados', {
            method: 'POST',
            body: formData
        });
        let result = await response.json();

        if (!result.job_id) throw new Error(result.message || 'Failed to start sequential job');
        let jobId = result.job_id;

        // 2. Continue with the next bands
        for (let i = 1; i < sortedFiles.length; i++) {
            showLoadingOverlay(`Uploading band ${i+1}/${sortedFiles.length}...`);
            let continueForm = new FormData();
            continueForm.append('file', sortedFiles[i]);
            continueForm.append('sequential_mode', 'continue');
            continueForm.append('job_id', jobId);

            let continueResp = await fetch('/predict_mados', {
                method: 'POST',
                body: continueForm
            });
            let continueResult = await continueResp.json();
            if (continueResult.status !== 'processing') {
                throw new Error(continueResult.message || `Failed to upload band ${i+1}`);
            }
        }

        // 3. Finish: trigger processing
        showLoadingOverlay('Finalizing and starting processing...');
        let finishForm = new FormData();
        finishForm.append('sequential_mode', 'finish');
        finishForm.append('job_id', jobId);

        let finishResp = await fetch('/predict_mados', {
            method: 'POST',
            body: finishForm
        });
        let finishResult = await finishResp.json();

        if (finishResult.status === 'processing' && finishResult.check_url) {
            // Poll for result
            const finalResult = await pollJobStatus(finishResult.check_url);
            // Display result
            displayMadosResult(finalResult);
            showSuccess('Sequential MADOS processing completed!');
            hideLoadingOverlay(); // Pastikan overlay ditutup setelah hasil ditampilkan
        } else {
            throw new Error(finishResult.message || 'Failed to start processing');
        }

    } catch (error) {
        hideLoadingOverlay();
        showError('Sequential MADOS error: ' + error.message);
        console.error(error);
    }
}

// Helper function to display MADOS results
function displayMadosResult(result) {
    try {
        // Display original image
        const originalImage = document.getElementById('originalOpticImage');
        if (result.original_image) {
            originalImage.src = result.original_image;
            originalImage.style.display = 'block';
        }

        // Display segmentation result
        const segmentationImage = document.getElementById('madosSegmentationImage');
        if (result.segmentation_image) {
            segmentationImage.src = result.segmentation_image;
            segmentationImage.style.display = 'block';
        }

        // Store oil coordinates for OpenOil simulation
        if (result.oil_coordinates && result.oil_coordinates.length > 0) {
            window.madosOilCoordinates = result.oil_coordinates;
            window.madosDetectionResult = result; // Store the full result for simulation
            // Enable Optical OpenOil simulation with detected oil
            const useMadosCheckbox = document.getElementById('useMadosDetectedOil');
            if (useMadosCheckbox) {
                useMadosCheckbox.disabled = false;
                useMadosCheckbox.checked = true; // Auto-enable if oil detected
            }
            showSuccess(`Detected ${result.oil_coordinates.length} oil spill points`);
        } else {
            window.madosOilCoordinates = null;
            window.madosDetectionResult = null;
            const useMadosCheckbox = document.getElementById('useMadosDetectedOil');
            if (useMadosCheckbox) {
                useMadosCheckbox.disabled = true;
                useMadosCheckbox.checked = false;
            }
            showError('No oil spill detected in the image');
        }

        // Update map with bounds and overlay images if available
        if (result.bounds) {
            const bounds = [
                [result.bounds.south, result.bounds.west],
                [result.bounds.north, result.bounds.east]
            ];
            // Tambahkan/Update layer optic di Leaflet
            if (result.original_image) {
                if (window.opticLayer) {
                    window.map.removeLayer(window.opticLayer);
                }
                window.opticLayer = L.imageOverlay(result.original_image, bounds, { opacity: 1 }).addTo(window.map);
            }
            // Tambahkan/Update layer MADOS di Leaflet
            if (result.segmentation_image) {
                if (window.madosLayer) {
                    window.map.removeLayer(window.madosLayer);
                }
                window.madosLayer = L.imageOverlay(result.segmentation_image, bounds, { opacity: 0.8 }).addTo(window.map);
            }
            // Daftarkan layer ke window.layers agar layer control berfungsi
            window.layers = window.layers || {};
            if (window.opticLayer) window.layers.optic = window.opticLayer;
            if (window.madosLayer) window.layers.mados = window.madosLayer;
            window.map.fitBounds(bounds);
        }

        // Show success message
        showSuccess('MADOS processing completed successfully');
    } catch (error) {
        console.error('Error displaying MADOS result:', error);
        showError('Error displaying MADOS result: ' + error.message);
    }
}

// Function to toggle dropdown
function toggleDropdown(element) {
    const content = element.nextElementSibling;
    const toggle = element.querySelector('.dropdown-toggle');
    
    // Toggle active class on content
    content.classList.toggle('active');
    
    // Toggle active class on dropdown toggle
    toggle.classList.toggle('active');
    
    // Update visibility and height with smooth transition
    if (content.classList.contains('active')) {
        content.style.height = content.scrollHeight + 'px';
        content.style.opacity = '1';
        content.style.visibility = 'visible';
    } else {
        content.style.height = '0';
        content.style.opacity = '0';
        content.style.visibility = 'hidden';
    }
}

// Initialize layer controls with dropdowns
function initializeLayerControls() {
    // Set initial states for layer toggles
    const layerTypes = ['raster', 'segmentation', 'optic', 'mados'];
    layerTypes.forEach(type => {
        const button = document.getElementById(`${type}Toggle`);
        if (button) {
            button.classList.add('active');
        }
    });
    
    // Set initial states for oil boom toggles
    const boomTypes = ['Collection_Boom_Case_1', 'Diversion_Boom_Case_1', 'Protection_Boom_Case_1', 'Gudang_Oil_Boom'];
    boomTypes.forEach(type => {
        const button = document.getElementById(`${type.toLowerCase().replace(/_/g, '')}Toggle`);
        if (button) {
            button.classList.remove('active');
        }
    });
    
    // Initialize dropdowns
    const dropdowns = document.querySelectorAll('.layer-control-panel .control-group h4');
    dropdowns.forEach(dropdown => {
        const content = dropdown.nextElementSibling;
        const toggle = dropdown.querySelector('.dropdown-toggle');
        
        // Set initial state
        content.classList.add('active');
        toggle.classList.add('active');
        content.style.height = content.scrollHeight + 'px';
        content.style.opacity = '1';
        content.style.visibility = 'visible';
    });

    // Ensure proper positioning of time slider
    const timeSliderContainer = document.getElementById('timeSliderContainer');
    if (timeSliderContainer) {
        timeSliderContainer.style.display = 'none';
    }
}

function updateFileNotification(inputId, notificationId) {
    const input = document.getElementById(inputId);
    const notification = document.getElementById(notificationId);
    
    if (input.files.length > 0) {
        if (input.multiple) {
            // For multiple files (optical input)
            const fileNames = Array.from(input.files).map(file => file.name).join(', ');
            notification.textContent = `Selected files: ${fileNames}`;
        } else {
            // For single file (SAR input)
            notification.textContent = `Selected file: ${input.files[0].name}`;
        }
        notification.style.display = 'block';
    } else {
        notification.textContent = '';
        notification.style.display = 'none';
    }
}

// Function to load oil types for MADOS OpenOil
async function loadMadosOilTypes() {
    try {
        const response = await fetch('/available_oil_types');
        const data = await response.json();
        
        if (data.success) {
            const select = document.getElementById('madosOilType');
            select.innerHTML = '';
            
            data.oil_types.forEach(type => {
                const option = document.createElement('option');
                option.value = type;
                option.textContent = type;
                select.appendChild(option);
            });
        }
    } catch (error) {
        console.error('Error loading oil types:', error);
    }
}

// Function to simulate OpenOil using MADOS detection results
async function simulateMadosOpenOil() {
    try {
        showLoadingOverlay('Starting Optical OpenOil simulation...');
        
        // Get form values
        const oilType = document.getElementById('madosOilType').value;
        const radius = parseInt(document.getElementById('madosRadius').value);
        const numParticles = parseInt(document.getElementById('madosNumParticles').value);
        const durationHours = parseInt(document.getElementById('madosDurationHours').value || 24);
        const startTime = document.getElementById('madosStartTime').value;
        
        // Validate start time
        if (!startTime) {
            showError('Please enter a start time for the simulation');
            hideLoadingOverlay();
            return;
        }
        
        // Check if we should use detected oil coordinates
        const useMadosDetectedOil = document.getElementById('useMadosDetectedOil').checked;
        
        if (useMadosDetectedOil && window.madosOilCoordinates && window.madosOilCoordinates.length > 0) {
            // Use detected oil coordinates from MADOS
            const formData = new FormData();
            formData.append('oil_type', oilType);
            formData.append('radius', radius);
            formData.append('num_particles', numParticles);
            formData.append('duration_hours', durationHours);
            if (startTime) {
                formData.append('start_time', startTime);
            }
            formData.append('oil_coordinates', JSON.stringify(window.madosOilCoordinates));
            
            // Send request to OpenOil service
            const response = await fetch('/mados_openoil_simulation', {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const result = await response.json();
            
            if (result.success) {
                displayOpenOilResult(result);
                showSuccess('Optical OpenOil simulation completed successfully');
            } else {
                throw new Error(result.error || 'Failed to run OpenOil simulation');
            }
        } else {
            showError('No oil coordinates available from MADOS detection. Please run MADOS detection first.');
        }
    } catch (error) {
        console.error('Error in Optical OpenOil simulation:', error);
        showError('Error in Optical OpenOil simulation: ' + error.message);
    } finally {
        hideLoadingOverlay();
    }
}

// Function to display OpenOil results
function displayOpenOilResult(result) {
    // Determine source: 'SAR' or 'Optical' (MADOS)
    const isOptical = result.metadata && (result.metadata.source === 'MADOS' || result.metadata.source === 'Optical');
    const sarContainer = document.getElementById('openoilResultContainerSAR');
    const opticContainer = document.getElementById('openoilResultContainerOptic');

    // Hide both first
    if (sarContainer) sarContainer.style.display = 'none';
    if (opticContainer) opticContainer.style.display = 'none';

    // Select target elements
    let trajImg, budgetImg, animImg, timeSliderDiv, infoDiv, infoText;
    if (isOptical) {
        if (opticContainer) opticContainer.style.display = 'block';
        trajImg = document.getElementById('opticOpenOilTrajectory');
        budgetImg = document.getElementById('opticOpenOilBudget');
        animImg = document.getElementById('opticOpenOilAnimation');
        timeSliderDiv = document.getElementById('opticTimeSliderContainer');
        infoDiv = document.getElementById('madosOpenOilInfo');
        infoText = document.getElementById('madosOpenOilText');
    } else {
        if (sarContainer) sarContainer.style.display = 'block';
        trajImg = document.getElementById('sarOpenOilTrajectory');
        budgetImg = document.getElementById('sarOpenOilBudget');
        animImg = document.getElementById('sarOpenOilAnimation');
        timeSliderDiv = document.getElementById('sarTimeSliderContainer');
        infoDiv = document.getElementById('sarOpenOilInfo');
        infoText = document.getElementById('sarOpenOilText');
    }

    // Set images if available
    if (trajImg && result.plots && result.plots.trajectory) {
        trajImg.src = 'data:image/png;base64,' + result.plots.trajectory;
        trajImg.style.display = 'block';
    }
    if (budgetImg && result.plots && result.plots.oil_budget) {
        budgetImg.src = 'data:image/png;base64,' + result.plots.oil_budget;
        budgetImg.style.display = 'block';
    }
    if (animImg && result.plots && result.plots.animation) {
        animImg.src = 'data:image/gif;base64,' + result.plots.animation;
        animImg.style.display = 'block';
    }

    // Add time slider if available
    if (timeSliderDiv && result.trajectory && result.trajectory.length > 0) {
        timeSliderDiv.innerHTML = '';
        addTimeSlider(result.trajectory, timeSliderDiv);
    }

    // Show simulation details in info box
    if (infoText && result.metadata) {
        safeSetInnerHTML(
            isOptical ? 'madosOpenOilText' : 'sarOpenOilText',
            `<strong>Simulation Details:</strong><br>
            - Source: ${result.metadata.source || (isOptical ? 'Optical' : 'SAR')}<br>
            - Oil Type: ${result.metadata.oil_type}<br>
            - Number of Particles: ${result.metadata.num_particles}<br>
            - Duration: ${result.metadata.duration_hours} hours<br>
            - Number of Oil Points: ${result.metadata.num_oil_points || '-'}`
        );
        safeSetDisplay(isOptical ? 'madosOpenOilInfo' : 'sarOpenOilInfo', 'block');
    }
    if (infoDiv) infoDiv.style.display = 'block';
}

// Function to show error message
function showError(message) {
    const errorDiv = document.createElement('div');
    errorDiv.className = 'alert alert-danger';
    errorDiv.textContent = message;
    
    // Determine which section to append to based on the current context
    const opticSection = document.querySelector('.optic-openoil-section');
    const sarSection = document.querySelector('.sar-openoil-section');
    
    // Append to the appropriate section
    if (opticSection) {
        opticSection.appendChild(errorDiv);
    } else if (sarSection) {
        sarSection.appendChild(errorDiv);
    } else {
        // Fallback to appending to the body if no specific section is found
        document.body.appendChild(errorDiv);
    }
    
    // Remove error message after 5 seconds
    setTimeout(() => {
        errorDiv.remove();
    }, 5000);
}

// Initialize MADOS oil types when the page loads
document.addEventListener('DOMContentLoaded', function() {
    loadMadosOilTypes();
});

// Utility functions to safely set innerHTML and display
function safeSetInnerHTML(id, html) {
    const el = document.getElementById(id);
    if (el) el.innerHTML = html;
}
function safeSetDisplay(id, display) {
    const el = document.getElementById(id);
    if (el) el.style.display = display;
}

// Function to show success message
function showSuccess(message) {
    const successDiv = document.createElement('div');
    successDiv.className = 'alert alert-success';
    successDiv.textContent = message;
    
    // Determine which section to append to based on the current context
    const opticSection = document.querySelector('.optic-openoil-section');
    const sarSection = document.querySelector('.sar-openoil-section');
    
    // Append to the appropriate section
    if (opticSection) {
        opticSection.appendChild(successDiv);
    } else if (sarSection) {
        sarSection.appendChild(successDiv);
    } else {
        // Fallback to appending to the body if no specific section is found
        document.body.appendChild(successDiv);
    }
    
    // Remove success message after 5 seconds
    setTimeout(() => {
        successDiv.remove();
    }, 5000);
}

// Add ESI layer variables
let esiLayers = {
    tipePantai: null,
    batasWilayah: null,
    spesies: null,
    kawasanLindung: null,
    penduduk: null
};

// Add ESI layer styles
const esiStyles = {
    tipePantai: {
        color: '#E64C00', // Default color for very high sensitivity
        weight: 2,
        opacity: 0.95,
        fillOpacity: 0.95,
        style: function(feature) {
            return {
                color: feature.properties.sensitivity === 'Tinggi' ? '#E69800' : '#E64C00',
                weight: 2,
                opacity: 0.95,
                fillOpacity: 0.95
            };
        }
    },
    batasWilayah: {
        color: '#0000FF',
        weight: 2,
        opacity: 0.5,
        // Remove fill color
        fillOpacity: 0
    },
    spesies: {
        color: '#F58356', // Default color for very high sensitivity
        weight: 2,
        opacity: 0.95,
        fillOpacity: 0.95,
        style: function(feature) {
            return {
                color: '#F58356', // Very high sensitivity
                weight: 2,
                opacity: 0.95,
                fillOpacity: 0.95
            };
        }
    },
    kawasanLindung: {
        color: '#D7191C', // Default color for very high sensitivity
        weight: 2,
        opacity: 0.95,
        fillOpacity: 0.95,
        style: function(feature) {
            const sensitivityColors = {
                'Sangat tinggi': '#D7191C',
                'Tinggi': '#FDAE61',
                'Sedang': '#FFFFBF',
                'Rendah': '#A6D96A',
                'Sangat rendah': '#1A9641'
            };
            return {
                color: sensitivityColors[feature.properties.kelas_sensitivitas] || '#D7191C',
                weight: 2,
                opacity: 0.95,
                fillOpacity: 0.95
            };
        }
    },
    penduduk: {
        color: '#592D00', // Default color for class 5
        weight: 2,
        opacity: 0.3,
        fillOpacity: 0.3,
        style: function(feature) {
            const classColors = {
                '1': '#FFF3E6',
                '2': '#D9A877',
                '3': '#B0773E',
                '4': '#85521F',
                '5': '#592D00'
            };
            return {
                color: classColors[feature.properties.kelas] || '#592D00',
                weight: 2,
                opacity: 0.3,
                fillOpacity: 0.3
            };
        }
    }
};

// Function to load ESI layers
async function loadESILayers() {
    try {
        // Load Tipe Pantai
        const tipePantaiResponse = await fetch('/static/data/ESI/TipePantai.geojson');
        const tipePantaiData = await tipePantaiResponse.json();
        esiLayers.tipePantai = L.geoJSON(tipePantaiData, {
            style: esiStyles.tipePantai.style,
            onEachFeature: function(feature, layer) {
                layer.bindPopup(`<b>Tipe Pantai:</b> ${feature.properties.TIPE_PANTAI || 'N/A'}<br>
                               <b>Sensitivity:</b> ${feature.properties.sensitivity || 'N/A'}`);
            }
        });

        // Load Batas Wilayah
        const batasWilayahResponse = await fetch('/static/data/ESI/BatasWilayah.geojson');
        const batasWilayahData = await batasWilayahResponse.json();
        esiLayers.batasWilayah = L.geoJSON(batasWilayahData, {
            style: esiStyles.batasWilayah,
            onEachFeature: function(feature, layer) {
                layer.bindPopup(`<b>Wilayah:</b> ${feature.properties.NAMA_WILAYAH || 'N/A'}`);
            }
        });

        // Load Spesies
        const spesiesResponse = await fetch('/static/data/ESI/Spesies.geojson');
        const spesiesData = await spesiesResponse.json();
        esiLayers.spesies = L.geoJSON(spesiesData, {
            style: esiStyles.spesies.style,
            onEachFeature: function(feature, layer) {
                layer.bindPopup(`<b>Spesies:</b> ${feature.properties.NAMA_SPESIES || 'N/A'}<br>
                               <b>Sensitivity:</b> ${feature.properties.kelas_sensitivitas || 'N/A'}`);
            }
        });

        // Load Kawasan Lindung
        const kawasanLindungResponse = await fetch('/static/data/ESI/KawasanLindung.geojson');
        const kawasanLindungData = await kawasanLindungResponse.json();
        esiLayers.kawasanLindung = L.geoJSON(kawasanLindungData, {
            style: esiStyles.kawasanLindung.style,
            onEachFeature: function(feature, layer) {
                layer.bindPopup(`<b>Kawasan:</b> ${feature.properties.NAMA_KAWASAN || 'N/A'}<br>
                               <b>Sensitivity:</b> ${feature.properties.kelas_sensitivitas || 'N/A'}`);
            }
        });

        // Load Penduduk
        const pendudukResponse = await fetch('/static/data/ESI/Penduduk.geojson');
        const pendudukData = await pendudukResponse.json();
        esiLayers.penduduk = L.geoJSON(pendudukData, {
            style: esiStyles.penduduk.style,
            onEachFeature: function(feature, layer) {
                layer.bindPopup(`<b>Kecamatan:</b> ${feature.properties.KECAMATAN || 'N/A'}<br>
                               <b>Jumlah Penduduk:</b> ${feature.properties.JUMLAH_PENDUDUK || 'N/A'}<br>
                               <b>Kelas:</b> ${feature.properties.kelas || 'N/A'}`);
            }
        });

        // Add layers to map in specified order
        const desiredOrder = [
            'batasWilayah', // Move batasWilayah to the beginning
            'tipePantai',
            'kawasanLindung',
            'spesies',
            'penduduk'
        ];

        desiredOrder.forEach(key => {
            const layer = esiLayers[key];
            if (layer) {
                layer.addTo(map);
            }
        });

        // Update layer controls
        updateESILayerControls();
    } catch (error) {
        console.error('Error loading ESI layers:', error);
        showError('Failed to load ESI layers');
    }
}

// Function to update ESI layer controls
function updateESILayerControls() {
    const esiControlGroup = document.querySelector('.esi-control-group');
    if (esiControlGroup) {
        const layerContent = esiControlGroup.querySelector('.layer-content');
        if (layerContent) {
            const layerButtons = layerContent.querySelector('.layer-buttons');
            if (layerButtons) {
                // Clear existing buttons
                layerButtons.innerHTML = '';

                // Add toggle buttons for each ESI layer
                Object.entries(esiLayers).forEach(([key, layer]) => {
                    if (layer) {
                        const button = document.createElement('button');
                        button.id = `${key}Toggle`;
                        button.className = 'btn btn-primary layer-dark';
                        button.onclick = () => toggleESILayer(key);
                        button.textContent = `Show ${key.replace(/([A-Z])/g, ' $1').trim()}`;
                        layerButtons.appendChild(button);

                        // Add opacity control
                        const opacityControl = document.createElement('div');
                        opacityControl.className = 'opacity-control';
                        opacityControl.innerHTML = `
                            <label for="${key}Opacity">${key.replace(/([A-Z])/g, ' $1').trim()} Opacity:</label>
                            <input id="${key}Opacity" type="range" min="0" max="100" value="80" 
                                   onchange="updateESILayerOpacity('${key}', this.value)">
                        `;
                        layerButtons.appendChild(opacityControl);
                    }
                });
                
                // Add legend toggle buttons
                // Removed legend toggle buttons as legends are now always visible
                
            }
        }
    }
}

// Function to toggle ESI layer visibility
function toggleESILayer(layerKey) {
    const layer = esiLayers[layerKey];
    if (layer) {
        if (map.hasLayer(layer)) {
            map.removeLayer(layer);
            document.getElementById(`${layerKey}Toggle`).classList.remove('active');
        } else {
            map.addLayer(layer);
            document.getElementById(`${layerKey}Toggle`).classList.add('active');
        }
    }
}

// Function to update ESI layer opacity
function updateESILayerOpacity(layerKey, value) {
    const layer = esiLayers[layerKey];
    if (layer) {
        const opacity = value / 100;
        layer.setStyle({
            ...esiStyles[layerKey],
            opacity: opacity,
            fillOpacity: opacity * 0.5
        });
    }
}

// Call loadESILayers when the map is initialized
document.addEventListener('DOMContentLoaded', function() {
    // ... existing initialization code ...
    loadESILayers();
});
