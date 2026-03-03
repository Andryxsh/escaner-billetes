import { INVALID_RANGES } from './data.js';

const { createWorker } = Tesseract;

// DOM Elements
const video = document.getElementById('video');
const ocrLiveText = document.getElementById('ocrLiveText');
const scanStatus = document.getElementById('scanStatus');
const resultPopup = document.getElementById('resultPopup');
const resultCard = document.getElementById('resultCard');
const resultIcon = document.getElementById('resultIcon');
const resultTitle = document.getElementById('resultTitle');
const resultSeries = document.getElementById('resultSeries');
const manualInput = document.getElementById('manualInput');
const manualVerifyBtn = document.getElementById('manualVerifyBtn');
const denomBtns = document.querySelectorAll('.denom-btn');

// Flag to pause OCR updates when user is typing manually
let isUserTyping = false;

manualInput.onfocus = () => { isUserTyping = true; };
manualInput.onblur = () => { setTimeout(() => { isUserTyping = false; }, 2000); };
manualVerifyBtn.onclick = () => {
    handleManualVerification();
};

// State
let currentDenom = '10';
let isProcessing = false;
let lastResult = '';
let resultTimeout;
let workersPool = [];
let currentStream = null;
let currentFacingMode = 'environment';

// Voting System State
let frameVotes = new Map();
let lastEvaluatedFrame = Date.now();
const VOTE_THRESHOLD = 3; // Aumentado para mayor seguridad
const FRAME_THROTTLE = 60;
let cvReady = false;

// Esperar a que OpenCV esté listo
window.onOpenCvReady = () => {
    cvReady = true;
    console.log("OpenCV.js is ready.");
};

// Fallback por si el script es async y no dispara el evento global
const checkCV = setInterval(() => {
    if (typeof cv !== 'undefined' && cv.getBuildInformation) {
        cvReady = true;
        clearInterval(checkCV);
        console.log("OpenCV detectado manualmente.");
    }
}, 500);


// Configuración de cámara
async function startCamera() {
    // Check for Secure Context
    if (!window.isSecureContext) {
        console.error("Contexto no seguro detectado. getUserMedia requiere HTTPS o localhost.");
        updateStatus("ERROR DE SEGURIDAD", "HTTPS REQUERIDO PARA CELULAR", true);
        return;
    }

    try {
        if (currentStream) {
            currentStream.getTracks().forEach(track => track.stop());
        }
        const stream = await navigator.mediaDevices.getUserMedia({
            video: {
                facingMode: currentFacingMode,
                width: { ideal: 1920 },
                height: { ideal: 1080 }
            }
        });
        currentStream = stream;
        video.srcObject = stream;
        await video.play().catch(e => console.warn("Autoplay prevenido", e));
    } catch (err) {
        console.error("Error de cámara completo:", err);
        let errorMsg = "REVISE PERMISOS";
        if (err.name === 'NotAllowedError') errorMsg = "PERMISO DENEGADO";
        if (err.name === 'NotFoundError') errorMsg = "CÁMARA NO ENCONTRADA";
        if (err.name === 'NotReadableError') errorMsg = "CÁMARA OCUPADA";

        updateStatus("ACCESO DENEGADO", errorMsg, true);
    }
}

document.getElementById('toggleCamera').addEventListener('click', () => {
    currentFacingMode = currentFacingMode === 'environment' ? 'user' : 'environment';
    startCamera();
});

// Selector de denominación
denomBtns.forEach(btn => {
    btn.addEventListener('click', () => {
        denomBtns.forEach(b => {
            b.classList.remove('active', 'opacity-100');
            b.classList.add('opacity-40');
            b.querySelector('div').className = "w-16 h-10 rounded-lg flex items-center justify-center border-2 border-white/20 glass text-white font-black";
            b.querySelector('span').className = "text-[10px] font-bold text-slate-400 uppercase tracking-tighter";
        });

        btn.classList.add('active', 'opacity-100');
        btn.classList.remove('opacity-40');
        btn.querySelector('div').className = "w-16 h-10 rounded-lg flex items-center justify-center border-2 border-primary bg-primary/20 text-white font-black";
        btn.querySelector('span').className = "text-[10px] font-bold text-primary uppercase tracking-tighter";
        currentDenom = btn.dataset.denom;
        frameVotes.clear(); // Resetear votos al cambiar de billete
    });
});

// Inicialización de Tesseract (Pool de 2 Workers DUAL-CORE)
async function initWorker() {
    updateStatus("INICIANDO IA...", "PREPARANDO DUAL-CORE");

    const w1 = await createWorker('eng');
    await w1.setParameters({ tessedit_char_whitelist: '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ' });

    const w2 = await createWorker('eng');
    await w2.setParameters({ tessedit_char_whitelist: '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ' });

    workersPool = [
        { worker: w1, isBusy: false, id: 1 },
        { worker: w2, isBusy: false, id: 2 }
    ];

    updateStatus("ESPERANDO SERIE...", "COLOQUE EL BILLETE");
}

// Pipeline de Pre-procesamiento con OpenCV (Nivel Bancario)
function preProcessImage(canvas, context) {
    if (!cvReady) {
        // Fallback básico si CV aún no carga
        return basicPreProcess(canvas, context);
    }

    try {
        let src = cv.imread(canvas);
        let dst = new cv.Mat();

        // 1. Grayscale
        cv.cvtColor(src, src, cv.COLOR_RGBA2GRAY, 0);

        // 2. Bilateral Filter (Reduce ruido pero mantiene bordes de los números)
        cv.bilateralFilter(src, dst, 5, 75, 75, cv.BORDER_DEFAULT);

        // 3. Adaptive Thresholding (Clave para billetes viejos y sombras)
        cv.adaptiveThreshold(dst, dst, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2);

        // 4. Operaciones Morfológicas (Engrosar un poco los números si están borrados)
        let M = cv.Mat.ones(2, 2, cv.CV_8U);
        cv.morphologyEx(dst, dst, cv.MORPH_CLOSE, M);

        cv.imshow(canvas, dst);

        src.delete();
        dst.delete();
        M.delete();
    } catch (e) {
        console.error("Error en OpenCV pre-process:", e);
        basicPreProcess(canvas, context);
    }
}

function basicPreProcess(canvas, context) {
    const w = canvas.width;
    const h = canvas.height;
    const imageData = context.getImageData(0, 0, w, h);
    const data = imageData.data;
    for (let i = 0; i < data.length; i += 4) {
        const v = 0.2126 * data[i] + 0.7152 * data[i + 1] + 0.0722 * data[i + 2];
        const thresholdVal = v > 128 ? 255 : 0;
        data[i] = data[i + 1] = data[i + 2] = thresholdVal;
    }
    context.putImageData(imageData, 0, 0);
}

// Lógica de escaneo hiper-optimizada y concurrente
async function scanLoop() {
    // 1. Agendamos el próximo tick invariablemente para no bloquear el Event Loop
    scheduleNextFrame();

    if (workersPool.length === 0 || video.readyState < 2) {
        return;
    }

    // 2. Scheduler de Workers: Buscamos 1 motor libre
    const freeWorker = workersPool.find(w => !w.isBusy);
    if (!freeWorker) {
        return; // Todos los Workers saturados, descartamos frame
    }

    // Bloquear el worker seleccionado
    freeWorker.isBusy = true;

    // 3. Captura y Pre-procesamiento
    const canvas = document.createElement('canvas');
    const context = canvas.getContext('2d');

    const targetWidth = 300;
    const targetHeight = 80;
    const scale = video.videoWidth / video.clientWidth;

    const cropW = targetWidth * scale;
    const cropH = targetHeight * scale;
    const startX = (video.videoWidth - cropW) / 2;
    const startY = (video.videoHeight - cropH) / 2;

    canvas.width = targetWidth;
    canvas.height = targetHeight;

    context.drawImage(video, startX, startY, cropW, cropH, 0, 0, targetWidth, targetHeight);
    preProcessImage(canvas, context);

    // 4. Inferencia Paralela
    try {
        const { data: { text } } = await freeWorker.worker.recognize(canvas);
        const cleaned = text.replace(/\s/g, '');

        // Match serie (7-9 números) + letra de familia (A-Z)
        const match = cleaned.match(/(\d{7,9})([A-Z])?/i);

        if (match) {
            let numsOnly = match[1];
            const familyLetter = match[2] || '';

            // Forzar siempre 9 dígitos (rellenando con ceros a la izquierda)
            numsOnly = numsOnly.padStart(9, '0');

            const currentVotes = (frameVotes.get(numsOnly) || 0) + 1;
            frameVotes.set(numsOnly, currentVotes);

            if (currentVotes >= VOTE_THRESHOLD) {
                // Actualizar input manual solo si el usuario no está escribiendo
                if (!isUserTyping) {
                    manualInput.value = `${numsOnly}${familyLetter}`;
                }
                updateStatus("ESCANEO EXITOSO", `<span class="text-primary font-black animate-pulse">${numsOnly} <span class="text-slate-500">${familyLetter}</span></span>`, false, true);
                verifySeries(numsOnly, familyLetter);
                frameVotes.clear();
            } else {
                if (!isUserTyping) manualInput.value = `${numsOnly}${familyLetter}`;
                updateStatus("CONFIRMANDO...", "LEYENDO SERIE...");
            }
        } else {
            // Limpieza de caché si pasa mucho tiempo
            if (Date.now() - lastEvaluatedFrame > 500) frameVotes.clear();

            if (cleaned.length > 2 && /\d/.test(cleaned)) {
                updateStatus("AJUSTANDO...", "ACERQUE EL BILLETE");
            } else {
                updateStatus("ESPERANDO SERIE...", "COLOQUE EL BILLETE");
            }
        }
    } catch (e) {
        console.error(`OCR Error (Worker ${freeWorker.id}):`, e);
    } finally {
        // Liberar worker incondicionalmente al terminar su promesa
        freeWorker.isBusy = false;
        lastEvaluatedFrame = Date.now();
    }
}

function scheduleNextFrame() {
    if (window.requestIdleCallback) {
        requestIdleCallback(() => setTimeout(() => requestAnimationFrame(scanLoop), FRAME_THROTTLE));
    } else {
        setTimeout(() => requestAnimationFrame(scanLoop), FRAME_THROTTLE * 2);
    }
}

// Actualización de UI unificada
function updateStatus(title, subtitleHtml, isError = false, isSuccess = false) {
    if (isError) {
        scanStatus.className = "absolute -top-8 left-0 right-0 text-center text-[10px] uppercase tracking-[0.2em] font-black text-invalid-red";
        ocrLiveText.innerHTML = `<span class="text-invalid-red">${subtitleHtml}</span>`;
    } else if (isSuccess) {
        scanStatus.className = "absolute -top-8 left-0 right-0 text-center text-[10px] uppercase tracking-[0.2em] font-black text-valid-green";
        ocrLiveText.innerHTML = subtitleHtml;
    } else {
        scanStatus.className = "absolute -top-8 left-0 right-0 text-center text-[10px] uppercase tracking-[0.2em] font-black font-black text-primary pulse-text";
        ocrLiveText.innerHTML = subtitleHtml;
    }
    scanStatus.innerText = title;
}

function handleManualVerification() {
    const rawVal = manualInput.value.trim().toUpperCase();
    const match = rawVal.match(/(\d{1,9})([A-Z])?/);

    if (match) {
        let numsOnly = match[1].padStart(9, '0');
        const familyLetter = match[2] || '';
        manualInput.value = `${numsOnly}${familyLetter}`;
        verifySeries(numsOnly, familyLetter);
    } else {
        updateStatus("ERROR DE FORMATO", "REVISE EL NÚMERO", true);
    }
}

// Verificación contra la DB local
function verifySeries(seriesNumStr, familyLetter) {
    if (seriesNumStr === lastResult) return;
    lastResult = seriesNumStr;

    const seriesNum = parseInt(seriesNumStr, 10);
    const ranges = INVALID_RANGES[currentDenom];
    const isInvalid = ranges.some(range => seriesNum >= range[0] && seriesNum <= range[1]);

    showResult(isInvalid, `${seriesNumStr} ${familyLetter}`);
}

// Output final estructurado
function showResult(isInvalid, series) {
    if (window.navigator.vibrate) window.navigator.vibrate(isInvalid ? [200, 100, 200] : 50);

    resultPopup.classList.add('active');
    resultSeries.innerText = series;

    const baseIconClass = "size-16 rounded-full bg-white/20 flex items-center justify-center shrink-0";

    if (isInvalid) {
        resultCard.className = "bg-invalid-red rounded-3xl p-6 flex items-center gap-4 shadow-[0_20px_60px_rgba(239,68,68,0.4)]";
        resultTitle.innerText = "¡BILLETE INVÁLIDO!";
        resultIcon.className = baseIconClass;
        resultIcon.innerHTML = '<i class="fa-solid fa-triangle-exclamation text-3xl text-white"></i>';
    } else {
        resultCard.className = "bg-valid-green rounded-3xl p-6 flex items-center gap-4 shadow-[0_20px_60px_rgba(16,185,129,0.4)]";
        resultTitle.innerText = "BILLETE VÁLIDO";
        resultIcon.className = baseIconClass;
        resultIcon.innerHTML = '<i class="fa-solid fa-circle-check text-3xl text-white"></i>';
    }

    clearTimeout(resultTimeout);
    resultTimeout = setTimeout(() => {
        resultPopup.classList.remove('active');
        lastResult = '';
    }, 4000);
}

// Boot sequence
video.addEventListener('loadedmetadata', () => {
    scanLoop();
});

startCamera();
initWorker();
