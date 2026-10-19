import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const COLORS = { queued: 0xe8b867, processing: 0x56cbbd, completed: 0x729aec };

export function createScene(container, onSelect) {
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x111d2e);
  const camera = new THREE.PerspectiveCamera(40, 1, 0.1, 150);
  camera.position.set(11, 10, 13);
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.domElement.setAttribute('aria-label', '机器与作业的 3D 状态场景');
  container.appendChild(renderer.domElement);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.target.set(0, 0.5, 0);
  controls.enableDamping = true;
  controls.maxPolarAngle = Math.PI * 0.48;
  controls.minDistance = 5;
  controls.maxDistance = 60;
  scene.add(new THREE.HemisphereLight(0xe4f2ff, 0x263c55, 2.6));
  const light = new THREE.DirectionalLight(0xffffff, 3);
  light.position.set(5, 10, 7);
  scene.add(light);

  function box(size, position, color) {
    const mesh = new THREE.Mesh(new THREE.BoxGeometry(...size),
      new THREE.MeshStandardMaterial({ color, roughness: 0.6, metalness: 0.15 }));
    mesh.position.set(...position);
    scene.add(mesh);
    return mesh;
  }
  box([14, 0.2, 8], [0, -0.15, 0], 0x24364b);
  scene.add(new THREE.GridHelper(14, 14, 0x486077, 0x30445b));
  box([2, 0.5, 2.3], [0, 0.25, 0], 0x344b62);
  box([0.2, 2.2, 0.2], [-0.9, 1.3, -0.95], 0x698197);
  box([0.2, 2.2, 0.2], [0.9, 1.3, -0.95], 0x698197);
  box([2, 0.25, 0.3], [0, 2.35, -0.95], 0x56cbbd);
  box([4.2, 0.22, 2.8], [-3.8, 0.1, 0], 0x30415a);
  box([4.2, 0.22, 2.8], [3.8, 0.1, 0], 0x30415a);
  const activeLight = box([0.22, 0.22, 0.22], [0.9, 2.35, -0.75], 0x56cbbd);
  const jobs = new Map();
  const raycaster = new THREE.Raycaster();
  const pointer = new THREE.Vector2();
  let down;
  renderer.domElement.addEventListener('pointerdown', e => { down = [e.clientX, e.clientY]; });
  renderer.domElement.addEventListener('pointerup', e => {
    if (!down || Math.hypot(e.clientX - down[0], e.clientY - down[1]) > 5) return;
    const rect = renderer.domElement.getBoundingClientRect();
    pointer.set((e.clientX - rect.left) / rect.width * 2 - 1, -(e.clientY - rect.top) / rect.height * 2 + 1);
    raycaster.setFromCamera(pointer, camera);
    const hit = raycaster.intersectObjects([...jobs.values()].filter(m => m.visible))[0];
    if (hit) onSelect(hit.object.userData.jobId);
  });
  const resize = new ResizeObserver(() => {
    camera.aspect = container.clientWidth / container.clientHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(container.clientWidth, container.clientHeight);
  });
  resize.observe(container);
  renderer.setAnimationLoop(() => { controls.update(); renderer.render(scene, camera); });

  function clear() {
    for (const mesh of jobs.values()) {
      scene.remove(mesh); mesh.geometry.dispose(); mesh.material.dispose();
    }
    jobs.clear();
    activeLight.material.color.setHex(0x698197);
  }

  function setJobs(items) {
    clear();
    items.forEach(job => {
      const mesh = box([0.55, 0.55, 0.55], [0, 0.75, 0], COLORS.queued);
      mesh.userData.jobId = job.id;
      jobs.set(job.id, mesh);
    });
  }

  function update(states, selected) {
    let queued = 0, completed = 0;
    states.forEach(job => {
      const mesh = jobs.get(job.id);
      if (!mesh) return;
      mesh.visible = job.state !== 'not_arrived';
      mesh.material.color.setHex(COLORS[job.state] ?? COLORS.queued);
      mesh.material.emissive.setHex(job.id === selected ? 0x34465e : 0x000000);
      mesh.scale.setScalar(job.id === selected ? 1.15 : 1);
      if (job.state === 'processing') mesh.position.set(0, 0.85, 0);
      if (job.state === 'queued') {
        const rank = queued++;
        mesh.position.set(-2.5 - (rank % 5) * 0.7, 0.65 + Math.floor(rank / 20) * 0.65, (Math.floor(rank / 5) % 4 - 1.5) * 0.7);
      }
      if (job.state === 'completed') {
        const rank = completed++;
        mesh.position.set(2.5 + (rank % 5) * 0.7, 0.65 + Math.floor(rank / 20) * 0.65, (Math.floor(rank / 5) % 4 - 1.5) * 0.7);
      }
    });
    activeLight.material.color.setHex(states.some(j => j.state === 'processing') ? 0x56cbbd : 0x698197);
  }
  return { setJobs, update, clear };
}
