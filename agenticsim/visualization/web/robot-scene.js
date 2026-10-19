import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';

export function createRobotScene(container, onSelect) {
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0xdfe5eb);
  scene.fog = new THREE.Fog(0xdfe5eb, 7, 18);
  const camera = new THREE.PerspectiveCamera(37, 1, .02, 50);
  camera.up.set(0, 0, 1);
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = .86;
  renderer.domElement.setAttribute('aria-label', '真实机器人网格与 Newton 仿真姿态');
  container.append(renderer.domElement);
  const pmrem = new THREE.PMREMGenerator(renderer), room = new RoomEnvironment();
  const environment = pmrem.fromScene(room, .04);
  scene.environment = environment.texture;
  scene.environmentIntensity = .75;
  room.dispose(); pmrem.dispose();
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true; controls.minDistance = .8; controls.maxDistance = 9;
  controls.maxPolarAngle = Math.PI * .48;
  const ambient = new THREE.HemisphereLight(0xf6faff, 0x7c828b, .65); scene.add(ambient);
  const key = new THREE.DirectionalLight(0xfff3e5, 2.2); key.position.set(3, -4, 6);
  key.castShadow = true; key.shadow.mapSize.set(2048, 2048);
  key.shadow.camera.left = -2.5; key.shadow.camera.right = 2.5;
  key.shadow.camera.top = 2.5; key.shadow.camera.bottom = -2.5;
  key.shadow.camera.near = .1; key.shadow.camera.far = 15;
  key.shadow.normalBias = .015; key.shadow.bias = -.0001; scene.add(key);
  const rim = new THREE.DirectionalLight(0xbad5ff, 1); rim.position.set(-3, 2, 3); scene.add(rim);
  const floor = new THREE.Mesh(new THREE.PlaneGeometry(40, 40), new THREE.MeshStandardMaterial({ color: 0xa9b2bf, roughness: .82, metalness: .08 }));
  floor.position.z = -.009; floor.receiveShadow = true; scene.add(floor);
  const grid = new THREE.GridHelper(16, 32, 0x96a3b1, 0xc0c8d1);
  grid.rotation.x = Math.PI / 2; grid.position.z = -.008; grid.material.transparent = true; grid.material.opacity = .4; scene.add(grid);
  const stage = new THREE.Group(); scene.add(stage);
  const edges = new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.BoxGeometry(2.6, 2.6, .025)), new THREE.LineBasicMaterial({ color: 0x698599 }));
  edges.position.z = -.018; scene.add(edges);
  const bodies = [], meshes = [], geometryCache = new Map();
  const axes = new THREE.AxesHelper(.18); axes.visible = false; scene.add(axes);
  const pathMaterial = new THREE.LineBasicMaterial({ color: 0x138d92, transparent: true, opacity: .85 });
  let traceLine, spec, down, selected;
  const ray = new THREE.Raycaster(), pointer = new THREE.Vector2();
  renderer.domElement.addEventListener('pointerdown', e => { down = [e.clientX, e.clientY]; });
  renderer.domElement.addEventListener('pointerup', e => {
    if (!down || Math.hypot(e.clientX - down[0], e.clientY - down[1]) > 5) return;
    const r = renderer.domElement.getBoundingClientRect();
    pointer.set((e.clientX - r.left) / r.width * 2 - 1, -(e.clientY - r.top) / r.height * 2 + 1);
    ray.setFromCamera(pointer, camera);
    const hit = ray.intersectObjects(meshes)[0];
    if (hit) onSelect(hit.object.userData.body);
  });
  const resize = new ResizeObserver(() => {
    camera.aspect = container.clientWidth / container.clientHeight; camera.updateProjectionMatrix();
    renderer.setSize(container.clientWidth, container.clientHeight);
  }); resize.observe(container);
  renderer.setAnimationLoop(() => { controls.update(); renderer.render(scene, camera); });

  function clear() {
    stage.clear(); bodies.length = 0;
    meshes.forEach(m => m.material.dispose()); meshes.length = 0;
    if (traceLine) { scene.remove(traceLine); traceLine.geometry.dispose(); traceLine = undefined; }
    axes.visible = false; selected = undefined;
  }
  function resetCamera() {
    const humanoid = spec?.model_id === 'unitree-g1';
    camera.position.set(humanoid ? 1.8 : 1.65, humanoid ? -1.9 : -1.55, 1.25);
    controls.target.set(humanoid ? 0 : .15, 0, humanoid ? .7 : .55); controls.update();
  }
  async function load(run, fetchBytes) {
    clear(); spec = run.scene;
    await Promise.all(spec.meshes.map(async info => {
      if (geometryCache.has(info.sha256)) return;
      const bytes = await fetchBytes(info.file);
      if (bytes.byteLength !== info.vertex_count * 24) throw new Error('机器人网格大小不一致');
      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(bytes, 0, info.vertex_count * 3), 3));
      geometry.setAttribute('normal', new THREE.BufferAttribute(new Float32Array(bytes, info.vertex_count * 12, info.vertex_count * 3), 3));
      geometry.computeBoundingSphere(); geometryCache.set(info.sha256, geometry);
    }));
    spec.bodies.forEach(() => { const group = new THREE.Group(); bodies.push(group); stage.add(group); });
    spec.geometries.forEach(g => {
      const info = spec.meshes.find(m => m.id === g.mesh);
      const material = new THREE.MeshStandardMaterial({ color: new THREE.Color().setRGB(...g.rgba.slice(0, 3)),
        roughness: Math.max(.26, .65 - g.shininess), metalness: Math.min(.5, g.specular * .7), side: THREE.DoubleSide });
      const mesh = new THREE.Mesh(geometryCache.get(info.sha256), material);
      mesh.position.fromArray(g.position); mesh.quaternion.fromArray(g.quaternion_xyzw);
      mesh.castShadow = mesh.receiveShadow = true; mesh.userData.body = g.body;
      bodies[g.body].add(mesh); meshes.push(mesh);
    });
    const tracked = spec.bodies.indexOf(spec.model_id === 'unitree-g1' ? 'pelvis' : 'link7');
    if (tracked >= 0) {
      const points = run.body_poses.map(p => new THREE.Vector3(...p[tracked].slice(0, 3)));
      traceLine = new THREE.Line(new THREE.BufferGeometry().setFromPoints(points), pathMaterial); scene.add(traceLine);
    }
    resetCamera(); update(run.body_poses[0]);
  }
  function update(poses, body = selected) {
    selected = body;
    poses.forEach((p, i) => { bodies[i].position.fromArray(p); bodies[i].quaternion.fromArray(p.slice(3)); });
    meshes.forEach(m => m.material.emissive.setHex(m.userData.body === selected ? 0x173c47 : 0));
    axes.visible = selected !== undefined;
    if (axes.visible) { axes.position.copy(bodies[selected].position); axes.quaternion.copy(bodies[selected].quaternion); }
    renderer.domElement.dataset.bodyCount = String(bodies.length);
    renderer.domElement.dataset.triangleCount = String(spec.meshes.reduce((sum, m) => sum + m.vertex_count / 3, 0));
  }
  return { load, update, clear, resetCamera, setTrace: visible => { if (traceLine) traceLine.visible = visible; } };
}
