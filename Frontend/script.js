const { useState, useEffect, useRef } = React;

const BACKEND_URL = "http://localhost:8000";

// --- CONFIGURATION ---
const GLASSES_DB = [
    { id: "Aviator", src: "./frames/aviator.png", name: "Ray-Ban Aviator", price: 150, color: "bg-blue" },
    { id: "Wayfarer", src: "./frames/wayfarer.png", name: "Classic Wayfarer", price: 130, color: "bg-purple" },
    { id: "Round Gold", src: "./frames/round_golden.png", name: "Golden Round", price: 145, color: "bg-green" },
    { id: "Square Thick", src: "./frames/square_thick.png", name: "Bold Square", price: 125, color: "bg-orange" },
    { id: "Black Gold", src: "./frames/BlackGold_round.png", name: "Black & Gold", price: 160, color: "bg-blue" },
    { id: "Black White", src: "./frames/BlackWhite_round.png", name: "Monochrome", price: 140, color: "bg-purple" },
    { id: "CatEye Tortoise", src: "./frames/CatEye_tortoiseshell.png", name: "Tortoise Cat-Eye", price: 155, color: "bg-green" },
    { id: "CatEye White", src: "./frames/CatEye_white.png", name: "White Cat-Eye", price: 150, color: "bg-orange" },
    { id: "Classic Round", src: "./frames/Classic_Tortoiseshell_Round_Eyeglasses.png", name: "Classic Tortoise", price: 135, color: "bg-blue" },
    { id: "Round Thin", src: "./frames/Round_thin.png", name: "Minimalist Round", price: 120, color: "bg-purple" },
    { id: "Sun Round", src: "./frames/Sunglasses_Tortoiseshell_Round.png", name: "Tortoise Sun", price: 170, color: "bg-green" },
    { id: "Sun Aviator", src: "./frames/sunglasses_aviator.png", name: "Dark Aviator", price: 165, color: "bg-orange" },
    { id: "Square Tortoise", src: "./frames/tortoiseshellWhite_square.png", name: "Two-Tone Square", price: 145, color: "bg-blue" }
];

const App = () => {
    // --- STATE ---
    const [mode, setMode] = useState('live'); 
    const [backendStatus, setBackendStatus] = useState('checking');
    const [faceShape, setFaceShape] = useState("--");
    const [recommendations, setRecommendations] = useState([]);
    
    const [activeGlassId, setActiveGlassId] = useState("Aviator");
    const [activeTab, setActiveTab] = useState("all"); 
    
    const [adjustments, setAdjustments] = useState({ scale: 1.0, x: 0, y: 0 });
    const [pdList, setPdList] = useState([]); 
    
    const [uploadedImage, setUploadedImage] = useState(null);
    const [compareImage, setCompareImage] = useState(null); // Compare state
    
    // CART STATE
    const [cart, setCart] = useState([]);
    const [isCartOpen, setIsCartOpen] = useState(false);
    const [showBuyModal, setShowBuyModal] = useState(false);

    // --- REFS ---
    const activeGlassRef = useRef("Aviator");
    const loadedAssetsRef = useRef({});
    const adjustmentsRef = useRef({ scale: 1.0, x: 0, y: 0 }); 
    const lastAnalysisTimeRef = useRef(0);
    const lastFrameTimeRef = useRef(0); 
    
    const videoRef = useRef(null);
    const canvasRef = useRef(null);
    const faceMeshRef = useRef(null);
    const cameraRef = useRef(null);
    const fileInputRef = useRef(null);

    // --- 1. Image Preloading ---
    useEffect(() => {
        const loadAllImages = async () => {
            const assets = {};
            const promises = GLASSES_DB.map(glass => {
                return new Promise((resolve) => {
                    const img = new Image();
                    img.src = glass.src;
                    img.onload = () => { assets[glass.id] = img; resolve(); };
                    img.onerror = () => { console.error(`❌ Error loading: ${glass.id}`); resolve(); };
                });
            });
            await Promise.all(promises);
            loadedAssetsRef.current = assets;
        };
        loadAllImages();
    }, []);

    const handleGlassSelect = (id) => {
        setActiveGlassId(id);
        activeGlassRef.current = id; 
        if (mode === 'upload' && uploadedImage) faceMeshRef.current.send({image: uploadedImage});
    };

    useEffect(() => { adjustmentsRef.current = adjustments; if (mode === 'upload' && uploadedImage) faceMeshRef.current.send({image: uploadedImage}); }, [adjustments]);

    useEffect(() => {
        fetch(BACKEND_URL)
            .then(res => res.ok ? setBackendStatus('online') : setBackendStatus('offline'))
            .catch(() => setBackendStatus('offline'));
    }, []);

    // --- 3. Initialize MediaPipe (Optimized) ---
    useEffect(() => {
        const faceMesh = new FaceMesh({locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`});
        
        faceMesh.setOptions({ 
            maxNumFaces: 4, 
            refineLandmarks: true, 
            minDetectionConfidence: 0.5, 
            minTrackingConfidence: 0.5 
        });
        
        faceMesh.onResults(onResults);
        faceMeshRef.current = faceMesh;

        if(mode === 'live' && videoRef.current) {
            const camera = new Camera(videoRef.current, {
                onFrame: async () => { 
                    const now = Date.now();
                    if (now - lastFrameTimeRef.current > 30) {
                        lastFrameTimeRef.current = now;
                        await faceMesh.send({image: videoRef.current});
                    }
                },
                width: 640, height: 480
            });
            camera.start();
            cameraRef.current = camera;
        } else {
            if(cameraRef.current) cameraRef.current.stop();
        }
        return () => { if(cameraRef.current) cameraRef.current.stop(); }
    }, [mode]);

    // --- 4. Render Loop ---
    const onResults = (results) => {
        const canvas = canvasRef.current;
        if(!canvas) return;
        const ctx = canvas.getContext('2d');
        
        // Resize canvas to match input image (critical for upload mode)
        if (canvas.width !== results.image.width || canvas.height !== results.image.height) {
            canvas.width = results.image.width;
            canvas.height = results.image.height;
        }

        const now = Date.now();
        if (mode === 'live' && now - lastAnalysisTimeRef.current > 2000) {
            lastAnalysisTimeRef.current = now;
            performLiveAnalysis(results.image);
        }

        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.save();
        
        if (mode === 'live') {
            ctx.scale(-1, 1);
            ctx.translate(-canvas.width, 0);
        }

        ctx.drawImage(results.image, 0, 0, canvas.width, canvas.height);

        if (results.multiFaceLandmarks && results.multiFaceLandmarks.length > 0) {
            const currentPdValues = [];
            
            // Iterate through ALL detected faces
            results.multiFaceLandmarks.forEach((landmarks) => {
                // 1. Calculate PD for THIS specific face
                const val = calculateSinglePD(landmarks, results.image.width, results.image.height);
                if (val && !isNaN(val)) {
                    currentPdValues.push(val);
                }
                
                // 2. Draw glasses for THIS specific face
                const currentGlass = activeGlassRef.current;
                const assets = loadedAssetsRef.current;
                if (currentGlass && assets[currentGlass]) {
                    renderGlassesImage(ctx, landmarks, assets[currentGlass]);
                }
            });

            // FORCE UPDATE PD LIST
            if (mode === 'upload' || Math.random() > 0.8) {
                 setPdList(currentPdValues);
            }
        } else { 
            setPdList([]); 
        }
        
        ctx.restore();
    };

    // --- FIX: SAFE NUMBER CALCULATION ---
    const calculateSinglePD = (landmarks, width, height) => {
        const lPupil = landmarks[468]; const rPupil = landmarks[473];
        const lIrisLeft = landmarks[474]; const lIrisRight = landmarks[476];
        const dx = (lPupil.x - rPupil.x) * width; const dy = (lPupil.y - rPupil.y) * height;
        const pupilDistPx = Math.sqrt(dx*dx + dy*dy);
        const idx = (lIrisLeft.x - lIrisRight.x) * width; const idy = (lIrisLeft.y - lIrisRight.y) * height;
        const irisDiamPx = Math.sqrt(idx*idx + idy*idy);
        const mmPerPx = 11.7 / irisDiamPx;
        
        // Use Math.round trick to keep 1 decimal place as a NUMBER type
        const pd = Math.round(pupilDistPx * mmPerPx * 10) / 10;
        return pd;
    };

    const performLiveAnalysis = (imageElement) => {
        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = 640;
        tempCanvas.height = 480;
        const tCtx = tempCanvas.getContext('2d');
        tCtx.drawImage(imageElement, 0, 0, 640, 480);
        tempCanvas.toBlob(async (blob) => {
            if (!blob) return;
            try {
                const fd = new FormData();
                fd.append('file', blob, "live_capture.jpg");
                const res = await fetch(`${BACKEND_URL}/analyze_face`, { method: 'POST', body: fd });
                if(res.ok) {
                    const data = await res.json();
                    setFaceShape(data.face_shape);
                    setRecommendations(data.recommended_frames);
                    setBackendStatus('online');
                }
            } catch (err) { setBackendStatus('offline'); }
        }, 'image/jpeg', 0.8);
    };

    const renderGlassesImage = (ctx, landmarks, imgObj) => {
        const w = ctx.canvas.width; const h = ctx.canvas.height; const adj = adjustmentsRef.current;
        const leftEye = landmarks[33]; const rightEye = landmarks[263];
        const leftTemple = landmarks[234]; const rightTemple = landmarks[454]; const noseBridge = landmarks[168];
        const lx = leftEye.x * w, ly = leftEye.y * h; const rx = rightEye.x * w, ry = rightEye.y * h;
        const tx_l = leftTemple.x * w, ty_l = leftTemple.y * h; const tx_r = rightTemple.x * w, ty_r = rightTemple.y * h;
        const nx = noseBridge.x * w, ny = noseBridge.y * h;
        const faceWidth = Math.sqrt(Math.pow(tx_r - tx_l, 2) + Math.pow(ty_r - ty_l, 2));
        const angle = Math.atan2(ry - ly, rx - lx);
        const scaleWidth = (faceWidth * 1.1) * adj.scale; 
        const imgAspect = imgObj.width / imgObj.height; const scaleHeight = scaleWidth / imgAspect;
        ctx.save(); ctx.translate(nx + adj.x, ny + adj.y); ctx.rotate(angle);
        if(mode === 'live') ctx.scale(-1, 1);
        ctx.drawImage(imgObj, -scaleWidth / 2, -scaleHeight / 2 + (scaleHeight * 0.1), scaleWidth, scaleHeight);
        ctx.restore();
    };

    // --- UPLOAD HANDLER ---
    const handleUpload = async (e) => {
        const file = e.target.files[0];
        e.target.value = ''; // Allow re-selecting same file
        
        if (!file) return;
        
        // 1. Force Mode Switch
        setMode('upload');
        setUploadedImage(null); 
        setCompareImage(null); // Reset compare mode
        setFaceShape("Analyzing..."); 
        setPdList([]);
        
        console.log("File selected:", file.name);

        // 2. Analyze (Backend)
        const fd = new FormData(); fd.append('file', file);
        try {
            const res = await fetch(`${BACKEND_URL}/analyze_face`, {method: 'POST', body: fd});
            if (res.ok) { const data = await res.json(); setFaceShape(data.face_shape); setRecommendations(data.recommended_frames); setBackendStatus('online'); } 
            else { setFaceShape("Backend Error"); }
        } catch(err) { console.error("Upload analysis failed:", err); setFaceShape("Offline"); }
        
        // 3. Render (Frontend)
        const img = new Image();
        img.src = URL.createObjectURL(file);
        img.onload = async () => {
            console.log("Image loaded:", img.width, "x", img.height);
            setUploadedImage(img); 
            const canvas = canvasRef.current;
            if(canvas) {
                canvas.width = img.width;
                canvas.height = img.height;
                const ctx = canvas.getContext('2d');
                ctx.clearRect(0,0, canvas.width, canvas.height);
                ctx.drawImage(img, 0, 0);
            }
            await faceMeshRef.current.send({image: img});
        };
    };

    // --- CART FUNCTIONS ---
    const addToCart = () => {
        const item = GLASSES_DB.find(g => g.id === activeGlassId);
        if (item) {
            setCart([...cart, item]);
        }
    };

    const removeFromCart = (index) => {
        const newCart = [...cart];
        newCart.splice(index, 1);
        setCart(newCart);
    };

    const calculateTotal = () => {
        return cart.reduce((total, item) => total + item.price, 0);
    };

    const handleBuyNow = () => {
        setIsCartOpen(false);
        setShowBuyModal(true);
        setTimeout(() => {
            setCart([]);
            setShowBuyModal(false);
        }, 3000);
    };

    const takeSnapshot = () => {
        const canvas = canvasRef.current;
        if (!canvas) return;
        const dataUrl = canvas.toDataURL("image/png");
        const link = document.createElement('a');
        link.download = `vto-snapshot-${Date.now()}.png`;
        link.href = dataUrl;
        link.click();
    };

    const toggleCompare = () => {
        if (compareImage) {
            setCompareImage(null); 
        } else {
            const canvas = canvasRef.current;
            if (!canvas) return;
            setCompareImage(canvas.toDataURL("image/png")); 
        }
    };

    // --- SORTING ---
    const getFilteredFrames = () => {
        if (activeTab === 'rec') {
            return GLASSES_DB.filter(g => recommendations.includes(g.id));
        }
        return GLASSES_DB; 
    };

    const displayFrames = getFilteredFrames();
    const activeGlassDetails = GLASSES_DB.find(g => g.id === activeGlassId);

    // --- RENDER ---
    return (
        <div className="app-layout relative">
            {/* Navbar */}
            <nav className="navbar">
                <div className="flex items-center gap-2">
                    <span className="text-xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-purple-500">SpexUP</span>
                </div>
                <div className="flex gap-4 text-gray-400 items-center">
                    <i data-lucide="search" className="w-5 h-5 hover:text-white cursor-pointer"></i>
                    
                    {/* Cart Icon with Badge */}
                    <div className="relative cursor-pointer" onClick={() => setIsCartOpen(!isCartOpen)}>
                        <i data-lucide="shopping-bag" className="w-5 h-5 hover:text-white"></i>
                        {cart.length > 0 && (
                            <span className="absolute -top-2 -right-2 bg-red-500 text-white text-[10px] w-4 h-4 rounded-full flex items-center justify-center font-bold">
                                {cart.length}
                            </span>
                        )}
                    </div>
                </div>
            </nav>

            {/* CART MODAL (Slide-out) */}
            {isCartOpen && (
                <>
                    <div className="fixed inset-0 bg-black/50 z-40" onClick={() => setIsCartOpen(false)}></div>
                    <div className="fixed top-0 right-0 h-full w-80 bg-white z-50 shadow-2xl p-6 flex flex-col transform transition-transform animate-slide-in">
                        <div className="flex justify-between items-center mb-6">
                            <h2 className="text-xl font-bold text-gray-900">Your Cart</h2>
                            <i data-lucide="x" className="w-5 h-5 cursor-pointer text-gray-500" onClick={() => setIsCartOpen(false)}></i>
                        </div>
                        
                        <div className="flex-1 overflow-y-auto space-y-4">
                            {cart.length === 0 ? (
                                <div className="text-center text-gray-400 mt-10">Your cart is empty.</div>
                            ) : (
                                cart.map((item, idx) => (
                                    <div key={idx} className="flex items-center gap-3 p-2 border rounded-lg">
                                        <img src={item.src} className="w-12 h-8 object-contain" />
                                        <div className="flex-1">
                                            <div className="text-sm font-bold text-gray-800">{item.name}</div>
                                            <div className="text-xs text-gray-500">${item.price}</div>
                                        </div>
                                        <i data-lucide="trash-2" className="w-4 h-4 text-red-400 cursor-pointer hover:text-red-600" onClick={() => removeFromCart(idx)}></i>
                                    </div>
                                ))
                            )}
                        </div>

                        <div className="mt-4 pt-4 border-t">
                            <div className="flex justify-between font-bold text-lg mb-4 text-gray-900">
                                <span>Total:</span>
                                <span>${calculateTotal()}</span>
                            </div>
                            <button 
                                onClick={handleBuyNow}
                                disabled={cart.length === 0}
                                className="w-full bg-green-600 text-white py-3 rounded-lg font-bold hover:bg-green-700 disabled:opacity-50 disabled:cursor-not-allowed">
                                BUY NOW
                            </button>
                        </div>
                    </div>
                </>
            )}

            {/* BUY SUCCESS POPUP */}
            {showBuyModal && (
                <div className="fixed inset-0 flex items-center justify-center bg-black/80 z-50">
                    <div className="bg-white p-8 rounded-2xl text-center shadow-2xl animate-bounce-in">
                        <div className="w-16 h-16 bg-green-100 rounded-full flex items-center justify-center mx-auto mb-4">
                            <i data-lucide="check" className="w-8 h-8 text-green-600"></i>
                        </div>
                        <h2 className="text-2xl font-bold text-gray-900 mb-2">Order Placed!</h2>
                        <p className="text-gray-500">Processing payment...</p>
                    </div>
                </div>
            )}

            <div className="main-container">
                {/* 2. LEFT SIDEBAR */}
                <div className="left-sidebar">
                    <div className="info-card">
                        <div className="text-xs text-slate-400 uppercase font-bold tracking-wider mb-2">Face Analysis</div>
                        <div className="text-3xl font-light text-white mb-4">{faceShape}</div>
                        <div className="text-xs text-slate-400 uppercase font-bold tracking-wider mb-2">System Status</div>
                        <div className={`flex items-center text-[10px] font-bold ${backendStatus==='online'?'text-green-400':'text-red-400'}`}>
                            <span className={`w-2 h-2 rounded-full mr-2 ${backendStatus==='online'?'bg-green-500 shadow-[0_0_8px_#4ade80]':'bg-red-500'} animate-pulse`}></span>
                            {backendStatus === 'online' ? 'ONLINE' : 'OFFLINE'}
                        </div>
                    </div>

                    <div className="info-card flex-1">
                        <div className="flex justify-between items-center mb-6">
                            <span className="text-xs text-slate-400 uppercase font-bold tracking-wider">Fit Adjustments</span>
                            <button onClick={() => setAdjustments({scale: 1.0, x: 0, y: 0})} className="text-[10px] text-blue-400 hover:text-blue-300">RESET</button>
                        </div>
                        <div className="space-y-6">
                            <div>
                                <div className="flex justify-between text-[10px] text-slate-400 mb-2"><span>Scale</span><span>{Math.round(adjustments.scale * 100)}%</span></div>
                                <input type="range" min="0.5" max="1.5" step="0.05" value={adjustments.scale} onChange={(e) => setAdjustments({...adjustments, scale: parseFloat(e.target.value)})} />
                            </div>
                            <div>
                                <div className="flex justify-between text-[10px] text-slate-400 mb-2"><span>Vertical</span><span>{adjustments.y}px</span></div>
                                <input type="range" min="-100" max="100" step="5" value={adjustments.y} onChange={(e) => setAdjustments({...adjustments, y: parseInt(e.target.value)})} />
                            </div>
                            <div>
                                <div className="flex justify-between text-[10px] text-slate-400 mb-2"><span>Horizontal</span><span>{adjustments.x}px</span></div>
                                <input type="range" min="-50" max="50" step="5" value={adjustments.x} onChange={(e) => setAdjustments({...adjustments, x: parseInt(e.target.value)})} />
                            </div>
                        </div>
                    </div>
                </div>

                {/* 3. CENTER STAGE */}
                <div className="center-stage">
                    <div className="ar-frame">
                        <div className="mode-pill">
                            <div className={`pill-option ${mode==='live'?'active':''}`} onClick={() => { setMode('live'); setUploadedImage(null); }}>Live</div>
                            <div className={`pill-option ${mode==='upload'?'active':''}`} onClick={() => fileInputRef.current.click()}>Upload</div>
                        </div>

                        {/* Video Layer */}
                        <video ref={videoRef} className={`absolute opacity-0 pointer-events-none`} autoPlay playsInline muted></video>
                        
                        {/* Comparison Container */}
                        <div className={`relative transition-all duration-300 ${compareImage ? 'flex w-full h-full gap-2 p-2' : 'w-full h-full'}`}>
                            {compareImage && (
                                <div className="flex-1 relative border border-slate-600 rounded-2xl overflow-hidden bg-black/80">
                                    <img src={compareImage} className="w-full h-full object-contain" />
                                    <div className="absolute top-4 left-4 bg-black/60 px-3 py-1 rounded-full text-xs text-white border border-white/20">Saved Look</div>
                                </div>
                            )}
                            <div className={`relative ${compareImage ? 'flex-1 border border-slate-600 rounded-2xl overflow-hidden' : 'w-full h-full'}`}>
                                <canvas ref={canvasRef} className={`w-full h-full object-contain ${mode==='live'?'mirror':''}`}></canvas>
                            </div>
                        </div>
                        
                        {/* Floating Action Bar */}
                        <div className="absolute bottom-6 left-1/2 transform -translate-x-1/2 flex gap-4 z-20">
                            <button onClick={takeSnapshot} className="p-3 bg-white text-black rounded-full shadow-lg hover:bg-gray-200 transition group" title="Take Snapshot">
                                <i data-lucide="camera" className="w-5 h-5"></i>
                            </button>
                            <button onClick={toggleCompare} className={`p-3 rounded-full shadow-lg transition ${compareImage ? 'bg-blue-600 text-white' : 'bg-slate-700 text-white hover:bg-slate-600'}`} title="Compare Mode">
                                <i data-lucide="split-square-horizontal" className="w-5 h-5"></i>
                            </button>
                        </div>

                        {/* Upload Input */}
                        <input type="file" ref={fileInputRef} className="hidden" accept="image/*" onChange={handleUpload} />
                        
                        {/* Upload Hint Overlay */}
                        {mode === 'upload' && !uploadedImage && (
                            <div className="absolute inset-0 flex items-center justify-center bg-black/80 z-10 pointer-events-none">
                                <div className="text-white text-center">
                                    <div className="text-4xl mb-2">📸</div>
                                    <div className="font-bold">Select a Photo</div>
                                </div>
                            </div>
                        )}
                    </div>
                </div>

                {/* 4. RIGHT SIDEBAR */}
                <div className="right-sidebar">
                    <h2 className="text-xl font-bold tracking-tight mb-6">STYLING STUDIO</h2>

                    <div className="bg-white/50 p-4 rounded-xl mb-6 flex justify-between items-center">
                        <div className="w-full">
                            <div className="text-sm font-bold text-gray-700 mb-2">Pupillary Distance (Est.)</div>
                            {pdList.length > 0 ? (
                                <div className="space-y-1">
                                    {pdList.map((pd, index) => (
                                        <div key={index} className="flex justify-between text-sm">
                                            <span className="text-gray-500">Person {index + 1}:</span>
                                            <span className="font-bold text-gray-900">{pd} mm</span>
                                        </div>
                                    ))}
                                </div>
                            ) : (
                                <div className="text-lg font-bold text-gray-900">--</div>
                            )}
                        </div>
                    </div>

                    <div className="tabs">
                        <div className={`tab-item ${activeTab==='all'?'active':''}`} onClick={()=>setActiveTab('all')}>All Frames</div>
                        <div className={`tab-item ${activeTab==='rec'?'active':''}`} onClick={()=>setActiveTab('rec')}>
                            Recommended
                            {recommendations.length > 0 && <span className="ml-1 w-2 h-2 bg-orange-500 rounded-full inline-block mb-1"></span>}
                        </div>
                    </div>

                    <div className="frames-grid flex-1">
                        {displayFrames.map(glass => (
                            <div key={glass.id} 
                                 className={`frame-card ${glass.color} ${activeGlassId===glass.id ? 'selected' : ''}`}
                                 onClick={() => handleGlassSelect(glass.id)}
                            >
                                <div className="h-16 w-full flex items-center justify-center mb-2">
                                    {loadedAssetsRef.current[glass.id] ? 
                                        <img src={glass.src} className="max-w-full max-h-full object-contain drop-shadow-md" /> 
                                        : <span className="text-xs">...</span>}
                                </div>
                                <div className="text-xs font-bold text-gray-800 text-center w-full truncate">{glass.name}</div>
                                <div className="text-[10px] text-gray-600 mt-1">${glass.price}</div>
                                
                                {recommendations.includes(glass.id) && (
                                    <div className="rec-badge" title="Recommended"></div>
                                )}
                            </div>
                        ))}
                    </div>

                    <div className="mt-auto pt-6 border-t border-black/5">
                        <div className="flex justify-between items-center mb-4">
                            <div>
                                <div className="text-lg font-bold">{activeGlassDetails?.name || "Select Frame"}</div>
                                <div className="text-sm text-gray-500">${activeGlassDetails?.price || "--"}</div>
                            </div>
                        </div>
                        <button 
                            onClick={addToCart}
                            className="w-full bg-orange-500 text-white font-bold py-3 rounded-lg shadow-lg shadow-orange-500/30 hover:bg-orange-600 transition active:scale-95">
                            ADD TO CART
                        </button>
                    </div>
                </div>
            </div>
        </div>
    );
};

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(<App />);

setTimeout(() => { lucide.createIcons(); }, 100);