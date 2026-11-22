// Realistic animal bodies: skinned glTF models (Creative Commons, from Sketchfab via the Objaverse mirror; credits in
// ANIMAL_MODELS and NOTICE) driven by the trained animal network (animalnet.js).
//
// Every driven bone is turned rigidly from its rest pose so that its piece of the body lines up with the network's
// joints, with the body's left axis fixing the twist. The spine and tail bend progressively between the network's
// pelvis and chest (and tail) frames, the neck and head keep the model's own shape, and a two-bone IK pass puts each
// wrist / ankle exactly where the network (after paw planting) put it. Limb lengths are measured on the model and fed
// back to the network's retargeting, so the network's paws and the model's paws coincide.
import { GLTFLoader } from '../vendor/three/addons/loaders/GLTFLoader.js';

const K = (s) => String(s).toLowerCase().replace(/[^a-z0-9]/g, '');
const DOG_HIP = 0.469, DOG_HIND = 0.448; // network rest: Hips height, hind leg chain (UpLeg..FootEnd)
const PAW_FLEX = 100 * Math.PI / 180, HIND_PLANT = 0.5; // lifted paws flex at most 100 deg; planted hind feet half-flat (the heel lifts)
const TWIST_MAX = 12 * Math.PI / 180, HEAD_ROLL_MAX = 10 * Math.PI / 180;
// head and neck relative to the chest (degrees) and the smoothing time constant (s)
const HEAD_LIMIT = { yaw: 40, up: 30, down: 50, roll: 10 }, NECK_LIMIT = { yaw: 25, up: 25, down: 40, roll: 6 }, LOOK_SMOOTH = 0.08;

// rig: bone names as written in the GLB (matched case- and punctuation-insensitively)
//   root    bone that receives the world translation (topmost driven bone; may be the pelvis itself)
//   pelvis  network Hips; spine: pelvis->chest bones; neck: neck bones (first one = network Neck); head + nose tip
//   front.L/R: [scapula|null, upper arm, forearm, metacarpus|null] + tip (toe tip, one name or several averaged)
//   hind.L/R:  [thigh, shin, metatarsus|null] + tip; tail: base to tip
// carriage: 'model' keeps the species' own neck/head carriage (moved by the network's neck motion, at most carriageLimit degrees),
//   default is the dog's; headLift (degrees) raises the dog's neck and head for species that carry the head higher
// style: speed (gait speed factor), cadence (step-rate factor on top of size), maxGait 'run' (no gallop), bound (paired legs: hopping),
//   legReach (network legs longer than the model's: straighter legs), metaStiff (0-1: lower leg pulled toward the body's down axis)
export const ANIMAL_MODELS = {
  dog: {
    name: 'Dog', file: 'dog.glb', hips: 0.40,
    credit: { title: 'Animated Dog, Shiba Inu', author: 'quander', license: 'CC BY 4.0', url: 'https://sketchfab.com/3d-models/9abfce885a834399b2c3ccaed51cd474' },
    rig: {
      root: 'COG_jnt.5_03', pelvis: 'pelvis_jnt.6_04', spine: ['spine_1_jnt.35_033', 'spine_2_jnt.36_034', 'chest_jnt.37_035'],
      neck: ['neck_base_jnt.38_036', 'neck_mid_jnt.39_037'], head: 'head_jnt.40_038', nose: ['L_nostril_jnt.70_066', 'R_nostril_jnt.69_065'],
      front: { L: [['L_clavicle_jnt.104_0100', 'L_shoulder_jnt.105_0101', 'L_elbow_jnt.106_0102', 'L_wrist_jnt.107_0103'], 'LF_mid_jnt.112_0108'],
        R: [['R_clavicle_jnt.114_0110', 'R_shoulder_jnt.115_0111', 'R_elbow_jnt.116_0112', 'R_wrist_jnt.117_0113'], 'RF_mid_jnt.122_0119'] },
      hind: { L: [['L_hip_jnt.24_022', 'L_knee_jnt.25_023', 'L_ankle_jnt.26_024'], 'LB_mid_jnt.31_029'],
        R: [['R_hip_jnt.14_012', 'R_knee_jnt.15_013', 'R_ankle_jnt.16_014'], 'RB_mid_jnt.21_019'] },
      tail: ['tail_1_jnt.7_05', 'tail_2_jnt.8_06', 'tail_3_jnt.9_07', 'tail_4_jnt.10_08', 'tail_5_jnt.11_09', 'tail_6_jnt.12_010', 'tail_7_jnt.13_011'],
    },
  },
  wolf: {
    name: 'Wolf', file: 'wolf.glb', hips: 0.72,
    credit: { title: 'Wolf', author: 'HiagoTadeu', license: 'CC BY 4.0', url: 'https://sketchfab.com/3d-models/9e7aceacef0f4486b605abdf1f62e4ac' },
    rig: {
      root: 'Root_01', pelvis: 'Spine_06_02', spine: ['Spine_05_03', 'Spine_04_04', 'Spine_03_05', 'Spine_02_06', 'Spine_01_07'],
      neck: ['Neck_08'], head: 'Head_09', nose: ['JawnUp_end_068'],
      front: { L: [[null, 'UperArmFront_L_022', 'LowerArmFront_L_023', 'WristFront_L_024'], 'Clawn_02_Front_L_end_080'],
        R: [[null, 'UperArmFront_R_031', 'LowerArmFront_R_032', 'WristFront_R_033'], 'Clawn_02_Front_R_end_076'] },
      hind: { L: [['UperArmBack_L_039', 'LowerArmBack_L_040', 'WristBack_L_041'], 'Clawn_02_Back_L_end_087'],
        R: [['UperArmBack_R_048', 'LowerArmBack_R_049', 'WristBack_R_050'], 'Clawn_02_Back_R_end_073'] },
      tail: ['Tail_01_056', 'Tail_02_057', 'Tail_03_058', 'Tail_04_059', 'Tail_05_060', 'Tail_06_061', 'Tail_07_062', 'Tail_08_063', 'Tail_09_064', 'Tail_10_065'],
    },
  },
  fox: {
    name: 'Fox', file: 'fox.glb', hips: 0.36, style: { cadence: 1.08 },
    credit: { title: 'Fox', author: 'alexanders823', license: 'CC BY 4.0', url: 'https://sketchfab.com/3d-models/4e37c303f8a241aaa2511bd9f3019273' },
    rig: {
      root: 'Fox__01', pelvis: 'Fox_ Pelvis_02', spine: ['Fox_ Spine_011', 'Fox_ Spine1_012'],
      neck: ['Fox_ Neck_013', 'Fox_ Neck1_014', 'Fox_ Neck2_015'], head: 'Fox_ Head_016', nose: null,
      front: { L: [[null, 'Fox_ L UpperArm_00', 'Fox_ L Forearm_019', 'Fox_ L Hand_020'], 'Fox_ L Finger0_end_034'],
        R: [[null, 'Fox_ R UpperArm_023', 'Fox_ R Forearm_024', 'Fox_ R Hand_025'], 'Fox_ R Finger0_end_036'] },
      hind: { L: [['Fox_ L Thigh_03', 'Fox_ L Calf_04', 'Fox_ L HorseLink_05'], 'Fox_ L Foot_end_033'],
        R: [['Fox_ R Thigh_07', 'Fox_ R Calf_08', 'Fox_ R HorseLink_09'], 'Fox_ R Foot_end_035'] },
      tail: ['Fox_ Tail_027', 'Fox_ Tail1_028', 'Fox_ Tail2_029', 'Fox_ Tail3_030', 'Fox_ Tail4_031', 'Fox_ Tail5_032'],
    },
  },
  cat: {
    name: 'Cat', file: 'cat.glb', hips: 0.25, style: { cadence: 1.1 },
    credit: { title: 'Cat - Walking', author: 'pamoan10', license: 'CC BY-NC 4.0', url: 'https://sketchfab.com/3d-models/827aa2c7951747599629700703050673' },
    rig: {
      root: 'CATRigHub001_01', pelvis: 'CATRigHub001_01', spine: ['CATRigSpine1_010', 'CATRigSpine7_011', 'CATRigSpine8_012', 'CATRigSpine4_013', 'CATRigSpine5_014', 'CATRigHub002_015'],
      neck: ['CATRigSpine6_016', 'CATRigSpine2_017', 'CATRigSpine3_018'], head: 'CATRigHub003_019', nose: null,
      front: { L: [[null, 'CATRigLArm1_021', 'CATRigLArm2_00', 'CATRigLArmPalm_022'], 'CATRigLPalmBone001_023'],
        R: [[null, 'CATRigRArm1_025', 'CATRigRArm2_026', 'CATRigRArmPalm_027'], 'CATRigRPalmBone001_028'] },
      hind: { L: [['CATRigLLeg1_02', 'CATRigLLeg2_03', 'CATRigLLegAnkle_04'], 'CATRigLAnkleBone001_05'],
        R: [['CATRigRLeg1_06', 'CATRigRLeg2_07', 'CATRigRLegAnkle_08'], 'CATRigRAnkleBone001_09'] },
      tail: ['CATRigTail1_029', 'CATRigTail2_030', 'CATRigTail3_031', 'CATRigTail4_032', 'CATRigTail5_033'],
    },
  },
  horse: {
    name: 'Horse', file: 'horse.glb', hips: 1.45, headLift: 25, style: { speed: 1.15, legReach: 1.08, metaStiff: 0.6 },
    credit: { title: 'Horse', author: 'alainsorazu', license: 'CC BY-SA 4.0', url: 'https://sketchfab.com/3d-models/bef74ed402cc40e996eea028668d3c1c' },
    rig: {
      root: 'spine.001_69', pelvis: 'spine.001_69', spine: ['spine.002_68', 'spine.003_54', 'spine.004_52', 'spine.005_51', 'spine.006_35'],
      neck: ['neck.001_32', 'neck.002_29', 'neck.003_26', 'neck.004_23'], head: 'head_18', nose: ['nose.L_10', 'nose.R_12'],
      front: { L: [['shoulder.L_41', 'upper_arm.L_40', 'forearm.L_39', 'forefoot.L_38'], 'f_hoof.L_36'],
        R: [['shoulder.R_48', 'upper_arm.R_47', 'forearm.R_46', 'forefoot.R_45'], 'f_hoof.R_43'] },
      hind: { L: [['thigh.L_59', 'lower_leg.L_58', 'hind_foot.L_57'], 'r_hoof.L_55'], R: [['thigh.R_65', 'lower_leg.R_64', 'hind_foot.R_63'], 'r_hoof.R_61'] },
      tail: ['tail.001_4', 'tail.002_3', 'tail.003_2', 'tail.004_1', 'tail.005_0'],
    },
  },
  bear: {
    name: 'Brown bear', file: 'bear.glb', hips: 0.95, style: { speed: 0.85, cadence: 0.92, legReach: 1.05 },
    credit: { title: 'Bear', author: 'Nyilonelycompany', license: 'CC BY 4.0', url: 'https://sketchfab.com/3d-models/ce0d5eb86cf5459bb6bd20244cb44b27' },
    rig: {
      root: 'spine.004_02', pelvis: 'spine.004_02', spine: ['spine.005_04', 'spine.006_05', 'spine.007_06', 'spine.008_07'],
      neck: ['spine.009_08', 'spine.010_09'], head: 'spine.011_010', nose: ['nose.003_015'],
      front: { L: [['shoulder.L_0101', 'front_thigh.L_0102', 'front_shin.L_0103', 'front_foot.L_0104'], 'f_middle.03.L_end_0212'],
        R: [['shoulder.R_0123', 'front_thigh.R_0124', 'front_shin.R_0125', 'front_foot.R_0126'], 'f_middle.03.R_end_0216'] },
      hind: { L: [['thigh.L_0148', 'shin.L_0149', 'foot.L_0150'], 'r_middle.03.L_end_0223'], R: [['thigh.R_0168', 'shin.R_0169', 'foot.R_0170'], 'r_middle.03.R_end_0227'] },
      tail: ['spine.003_03'],
    },
  },
  elephant: {
    name: 'Elephant', file: 'elephant.glb', hips: 2.3, style: { maxGait: 'run', speed: 0.8, cadence: 0.9, legReach: 1.1 },
    credit: { title: 'walking elephant', author: 'syaharanichan', license: 'CC BY 4.0', url: 'https://sketchfab.com/3d-models/2705e7483341479e8015d996cd766611' },
    rig: {
      root: 'Bone_Armature.001', pelvis: 'Bone.023_Armature.001', spine: ['Bone.020_Armature.001', 'Bone.019_Armature.001', 'Bone.018_Armature.001'],
      neck: ['Bone.001_Armature.001'], head: 'Bone.041_Armature.001', nose: ['Bone.002_Armature.001'],
      front: { L: [[null, 'Bone.036_Armature.001', 'Bone.038_Armature.001', null], 'Bone.040_Armature.001'],
        R: [[null, 'Bone.035_Armature.001', 'Bone.037_Armature.001', null], 'Bone.039_Armature.001'] },
      hind: { L: [['Bone.030_Armature.001', 'Bone.032_Armature.001', null], 'Bone.034_Armature.001'], R: [['Bone.029_Armature.001', 'Bone.031_Armature.001', null], 'Bone.033_Armature.001'] },
      tail: ['Bone.025_Armature.001', 'Bone.026_Armature.001', 'Bone.027_Armature.001', 'Bone.028_Armature.001'],
    },
  },
  tiger: {
    name: 'Tiger', file: 'tiger.glb', hips: 0.88, style: { speed: 0.95, cadence: 0.95 },
    credit: { title: 'Running Tiger', author: 'francescolima74', license: 'CC BY 4.0', url: 'https://sketchfab.com/3d-models/fc2c7fd74dd04e39935803b884c77258' },
    rig: {
      root: 'Bip01 Pelvis_01', pelvis: 'Bip01 Pelvis_01', spine: ['Bip01 Spine_02', 'Bip01 Spine1_03', 'Bip01 Spine2_04'],
      neck: ['Bip01 Neck_05'], head: 'Bip01 Head_06', nose: ['Bip01 HeadNub_07'],
      front: { L: [['Bip01 L Clavicle_017', 'Bip01 L UpperArm_018', 'Bip01 L Forearm_019', 'Bip01 L Hand_020'], 'Bip01 L Finger0Nub_022'],
        R: [['Bip01 R Clavicle_023', 'Bip01 R UpperArm_024', 'Bip01 R Forearm_025', 'Bip01 R Hand_026'], 'Bip01 R Finger0Nub_028'] },
      hind: { L: [['Bip01 L Thigh_029', 'Bip01 L Calf_030', 'Bip01 L HorseLink_031'], 'Bip01 L Toe0Nub_034'],
        R: [['Bip01 R Thigh_035', 'Bip01 R Calf_036', 'Bip01 R HorseLink_037'], 'Bip01 R Toe0Nub_040'] },
      tail: ['Bip01 Tail_041', 'Bip01 Tail1_042', 'Bip01 Tail2_043', 'Bip01 Tail3_044'],
    },
  },
  jaguar: {
    name: 'Jaguar', file: 'jaguar.glb', hips: 0.66,
    credit: { title: 'jaguar', author: 'Ear.Rodriguez', license: 'CC BY 4.0', url: 'https://sketchfab.com/3d-models/91c61c329d2a4668816f81f08dfcd492' },
    rig: {
      root: 'Root_M_05', pelvis: 'Root_M_05', spine: ['Spine1_M_06', 'Spine2_M_07', 'Spine3_M_08', 'Chest_M_09'],
      neck: ['Neck_M_010', 'Neck1_M_011', 'Neck2_M_012'], head: 'Head_M_013', nose: null,
      front: { L: [['Scapula_L_0143', 'Shoulder_L_0144', 'Elbow_L_0145', 'Wrist_L_0146'], 'Fingers2_L_0148'],
        R: [['Scapula_R_0111', 'Shoulder_R_0112', 'Elbow_R_0113', 'Wrist_R_0114'], 'Fingers2_R_0116'] },
      hind: { L: [['Hip_L_0220', 'Knee_L_0221', 'Ankle_L_0222'], 'Toes2_L_0224'], R: [['Hip_R_0186', 'Knee_R_0187', 'Ankle_R_0188'], 'Toes2_R_0189'] },
      tail: ['Tail0_M_0208', 'Tail7_M_0209', 'Tail8_M_0210', 'Tail9_M_0211', 'Tail10_M_0212', 'Tail11_M_0213', 'Tail12_M_0214', 'Tail13_M_0215', 'Tail14_M_0216', 'Tail15_M_0200', 'Tail16_M_0218'],
    },
  },
  rabbit: {
    name: 'Rabbit', file: 'rabbit.glb', hips: 0.17, style: { bound: true, cadence: 1.15 },
    credit: { title: 'Rabbit Rigged', author: 'FourthGreen', license: 'CC BY 4.0', url: 'https://sketchfab.com/3d-models/e7213589744d436b9d96e2dbb31198a5' },
    rig: {
      root: 'Root_M_02', pelvis: 'Root_M_02', spine: ['Spine1_M_011', 'Spine2_M_012', 'Chest_M_013'],
      neck: ['Neck_M_019'], head: 'Head_M_020', nose: ['NoseEnd_M_022'],
      front: { L: [['frontRump_L_032', 'frontHip_L_033', 'frontKnee_L_034', 'frontAnkle_L_035'], 'frontToes_L_036'],
        R: [['frontRump_R_014', 'frontHip_R_015', 'frontKnee_R_00', 'frontAnkle_R_016'], 'frontToes_R_017'] },
      hind: { L: [['backHip_L_038', 'backKnee_L_039', 'backAnkle_L_040'], 'backToes1_L_042'], R: [['backHip_R_03', 'backKnee_R_04', 'backAnkle_R_05'], 'backToes1_R_07'] },
      tail: ['Tail0_M_08', 'Tail1_M_09', 'Tail2_M_010'],
    },
  },
};

export async function loadAnimalModel(THREE, key, fetchBuffer) {
  const cfg = ANIMAL_MODELS[key];
  if (!cfg) return null;
  const buf = await fetchBuffer(`data/animal/models/${cfg.file}`);
  const gltf = await new GLTFLoader().parseAsync(buf, '');
  return new AnimalModel(THREE, gltf, cfg);
}

// Put the skeleton in its bind pose: the pose the skin was made in, where the mesh is undeformed. The node pose stored
// in a file is whatever frame the author exported (a lying fox, a galloping horse, mid-stride cats); rest directions,
// twists and toe angles measured on it were wrong, which bent paws sideways and twisted the skin. Each joint's bind
// rotation relative to its parent joint comes from the inverse bind matrices (IBM_parent * IBM_joint^-1), which does
// not depend on the files' unit and axis conventions; joint offsets keep their stored lengths. Returns the number of
// joints changed.
export function toBindPose(T, root, only = null) {
  root.updateMatrixWorld(true);
  const ibm = new Map();
  root.traverse((o) => { if (o.isSkinnedMesh) o.skeleton.bones.forEach((b, i) => { if (!ibm.has(b)) ibm.set(b, o.skeleton.boneInverses[i]); }); });
  const depth = (o) => { let d = 0; for (let p = o.parent; p; p = p.parent) d++; return d; };
  const p = new T.Vector3(), q = new T.Quaternion(), s = new T.Vector3(); let n = 0;
  for (const b of [...ibm.keys()].sort((x, y) => depth(x) - depth(y))) {
    if (!ibm.has(b.parent) || (only && !only.has(b))) continue; // top joint keeps its placement; unlisted bones their authored pose
    ibm.get(b.parent).clone().multiply(ibm.get(b).clone().invert()).decompose(p, q, s);
    if (b.quaternion.angleTo(q) > 1e-4) n++;
    b.quaternion.copy(q);
  }
  root.updateMatrixWorld(true);
  return n;
}
// Rotate a model so its body stands upright and faces +Z: spine (pelvis -> neck) forward, legs (hip -> toe tip) down.
function standUpright(T, scene, group, pelvis, neck, legs) {
  group.updateMatrixWorld(true);
  const P = (o) => o.getWorldPosition(new T.Vector3());
  const down = legs.reduce((acc, [a, tip]) => acc.add(P(tip).sub(P(a)).normalize()), new T.Vector3()).normalize();
  const fwd = P(neck).sub(P(pelvis)); fwd.addScaledVector(down, -fwd.dot(down)).normalize();
  const M = new T.Matrix4().makeBasis(new T.Vector3().crossVectors(down.clone().negate(), fwd), down.clone().negate(), fwd);
  const R = new T.Quaternion().setFromRotationMatrix(M).invert();
  scene.quaternion.premultiply(R); scene.position.applyQuaternion(R); group.updateMatrixWorld(true);
}

// Muscles, breathing and soft tissue for the skinned animals: every skin bone gets a displacement along the surface
// normal (metres) in a small per-bone float texture, read in the vertex shader before skinning. Driven per frame from
// the gait: loaded legs bulge (more when the joint below is flexed), the chest breathes (faster and deeper when
// running, panting at a sprint) and the belly rides the body's vertical bounce on a spring.
class AnimalBodyFX {
  constructor(T, scene, model) {
    this.T = T; this.m = model; this.enabled = true; this.phase = 0; this.effort = 0; this.jig = 0; this.jigV = 0; this.prevY = null; this.prevVy = 0;
    const cat = new Map(), put = (o, c) => { if (o && !cat.has(o)) cat.set(o, c); };
    for (const l of [model.front.L, model.front.R]) { put(l.bones[0], 'shoulder'); put(l.bones[1], 'upper'); put(l.bones[2], 'lower'); }
    for (const l of [model.hind.L, model.hind.R]) { put(l.bones[0], 'upper'); put(l.bones[1], 'lower'); }
    model.spine.forEach((o, i) => put(o, model.spineU[i] > 0.5 ? 'chest' : 'belly'));
    put(model.neck[0], 'chest'); put(model.pelvis, 'belly');
    this.cat = cat; this.limbs = [['front', 'L', 0], ['front', 'R', 1], ['hind', 'L', 2], ['hind', 'R', 3]];
    this.meshes = [];
    scene.traverse((o) => {
      if (!o.isSkinnedMesh || !o.material) return;
      // a bone belongs to the nearest driven bone at or above it; only categorised driven bones (legs, chest, belly) move
      const drivenSet = new Set(model.driven), bones = o.skeleton.bones;
      const owner = bones.map((b) => { for (let x = b; x; x = x.parent) if (drivenSet.has(x)) return cat.has(x) ? x : null; return null; });
      const data = new Float32Array(bones.length * 4), tex = new T.DataTexture(data, bones.length, 1, T.RGBAFormat, T.FloatType);
      tex.needsUpdate = true;
      const pv = model.pelvis, bi = bones.indexOf(pv) >= 0 ? bones.indexOf(pv) : 0;
      const k = new T.Vector3().setFromMatrixScale(bones[bi].matrixWorld.clone().multiply(o.skeleton.boneInverses[bi]).multiply(o.bindMatrix)).x;
      const uni = { uMuscleTex: { value: tex }, uMuscleUnit: { value: 1 / Math.max(1e-9, k) } };
      for (const mat of [].concat(o.material)) {
        mat.onBeforeCompile = (sh) => {
          Object.assign(sh.uniforms, uni);
          sh.vertexShader = 'uniform sampler2D uMuscleTex;\nuniform float uMuscleUnit;\n' + sh.vertexShader.replace('#include <begin_vertex>', `#include <begin_vertex>
          #ifdef USE_SKINNING
          { float d = skinWeight.x * texelFetch(uMuscleTex, ivec2(int(skinIndex.x), 0), 0).r + skinWeight.y * texelFetch(uMuscleTex, ivec2(int(skinIndex.y), 0), 0).r
                    + skinWeight.z * texelFetch(uMuscleTex, ivec2(int(skinIndex.z), 0), 0).r + skinWeight.w * texelFetch(uMuscleTex, ivec2(int(skinIndex.w), 0), 0).r;
            transformed += normal * d * uMuscleUnit; }
          #endif`);
        };
        mat.customProgramCacheKey = () => 'animal-body-fx';
        mat.needsUpdate = true;
      }
      this.meshes.push({ data, tex, owner });
    });
  }
  // contacts: paw contact weights (front L/R, hind L/R); speed in m/s
  update(contacts, speed, dt) {
    if (!(dt > 0)) return;
    const T = this.T, m = this.m, h = m.hips, val = new Map();
    if (this.enabled) {
      const effT = Math.min(1, speed / (4.5 * h + 0.5));
      this.effort += (effT - this.effort) * (1 - Math.exp(-dt / 0.6));
      const E = this.effort, rate = 0.35 + 1.9 * E * E; // breaths per second: 0.35 resting, ~2.2 panting
      this.phase = (this.phase + dt * rate * 2 * Math.PI) % (2 * Math.PI);
      const breath = (0.005 + 0.012 * E) * h * (0.5 - 0.5 * Math.cos(this.phase));
      // belly: spring driven by the pelvis' vertical acceleration (lags and overshoots the bounce)
      const y = m.pelvis.getWorldPosition(new T.Vector3()).y;
      if (this.prevY !== null) {
        const vy = (y - this.prevY) / dt, ay = Math.max(-60, Math.min(60, (vy - this.prevVy) / dt)); this.prevVy = vy;
        const w = 2 * Math.PI * 3.2, z = 0.35; // 3.2 Hz, lightly damped
        this.jigV += (-w * w * this.jig - 2 * z * w * this.jigV - ay) * dt; this.jig += this.jigV * dt;
        this.jig = Math.max(-0.03 * h, Math.min(0.03 * h, this.jig));
      }
      this.prevY = y;
      for (const [o, c] of this.cat) if (c === 'chest') val.set(o, breath); else if (c === 'belly') val.set(o, 0.6 * breath - 0.5 * this.jig);
      for (const [kind, s, k] of this.limbs) {
        const l = m[kind][s], b = l.bones.filter(Boolean), load = contacts ? Math.max(0, Math.min(1, (contacts[k] - 0.2) / 0.5)) : 0;
        const u = kind === 'front' ? b[b.length - 3] || b[0] : b[0], lo = kind === 'front' ? b[b.length - 2] : b[1], end = kind === 'front' ? b[b.length - 1] : b[2];
        let flex = 0;
        if (u && lo && end) { const P = (o) => o.getWorldPosition(new T.Vector3()); flex = Math.min(1, P(lo).sub(P(u)).angleTo(P(end).sub(P(lo))) / 1.6); }
        const bulge = h * (0.004 + 0.008 * load * (0.4 + 0.6 * flex) + 0.003 * E);
        if (u) val.set(u, bulge); if (lo) val.set(lo, 0.45 * bulge); if (kind === 'front' && l.bones[0]) val.set(l.bones[0], 0.6 * bulge);
      }
    }
    for (const me of this.meshes) {
      let changed = false;
      me.owner.forEach((o, i) => { const v = o ? val.get(o) || 0 : 0; if (me.data[i * 4] !== v) { me.data[i * 4] = v; changed = true; } });
      if (changed) me.tex.needsUpdate = true;
    }
  }
}

export class AnimalModel {
  constructor(THREE, gltf, cfg) {
    const T = THREE; this.T = T; this.cfg = cfg; this.group = new T.Group(); this.group.add(gltf.scene);
    const nodes = {}; gltf.scene.traverse((o) => { const k = K(o.name); if (k && !(k in nodes)) nodes[k] = o; });
    const B = (n) => { if (n == null) return null; const o = nodes[K(n)]; if (!o) throw new Error(`${cfg.name}: no bone ${n}`); return o; };
    const r = cfg.rig;
    this.root = B(r.root); this.pelvis = B(r.pelvis); this.spine = r.spine.map(B); this.neck = r.neck.map(B); this.head = B(r.head);
    this.noseB = r.nose ? [].concat(r.nose).map(B) : []; this.tail = (r.tail || []).map(B);
    const limb = ([bones, tip]) => ({ bones: bones.map(B), tip: [].concat(tip).map(B) });
    this.front = { L: limb(r.front.L), R: limb(r.front.R) }; this.hind = { L: limb(r.hind.L), R: limb(r.hind.R) };
    // reference pose: the skin's bind pose, standing upright (see toBindPose)
    // the driven chains, their ancestors and the toes down to the tips go to the bind pose; faces, ears and manes keep
    // the authored expression (some bind poses have the mouth wide open)
    const chainSet = new Set(), up = (o) => { for (let x = o; x && x !== gltf.scene; x = x.parent) chainSet.add(x); };
    [this.root, this.pelvis, ...this.spine, ...this.neck, this.head, ...this.tail].forEach((o) => o && up(o));
    for (const l of [this.front.L, this.front.R, this.hind.L, this.hind.R]) { l.bones.forEach((o) => o && up(o)); l.tip.forEach(up); }
    this.bindJoints = cfg.filePose ? 0 : toBindPose(T, gltf.scene, chainSet);
    const legRoots = [this.front.L, this.front.R, this.hind.L, this.hind.R].map((l) => [l.bones.find(Boolean), l.tip[0]]);
    if (!cfg.filePose) standUpright(T, gltf.scene, this.group, this.pelvis, this.neck[0], legRoots);
    gltf.scene.traverse((o) => { if (o.isMesh) { o.frustumCulled = false; o.castShadow = true; } });
    // scale: the model's hind leg is as long, relative to its hips height, as the dog's
    this.group.updateMatrixWorld(true);
    const P = (o) => o.getWorldPosition(new T.Vector3()), tipP = (l) => l.tip.reduce((a, o) => a.add(P(o)), new T.Vector3()).multiplyScalar(1 / l.tip.length);
    const chainLen = (l) => { const pts = [...l.bones.filter(Boolean).map(P), tipP(l)]; let s = 0; for (let i = 1; i < pts.length; i++) s += pts[i].distanceTo(pts[i - 1]); return s; };
    const hind0 = (chainLen(this.hind.L) + chainLen(this.hind.R)) / 2;
    this.hips = cfg.hips; this.s = cfg.hips * (DOG_HIND / DOG_HIP) / hind0;
    this.group.scale.setScalar(this.s); this.group.updateMatrixWorld(true);
    // rest pose (world, metres) of every driven bone
    this.driven = []; const add = (o) => { if (o && !this.driven.includes(o)) this.driven.push(o); };
    [this.root, this.pelvis, ...this.spine, ...this.neck, this.head, ...this.tail].forEach(add);
    for (const l of [this.front.L, this.front.R, this.hind.L, this.hind.R]) l.bones.forEach(add);
    const depth = (o) => { let d = 0; for (let p = o.parent; p; p = p.parent) d++; return d; };
    this.driven.sort((a, b) => depth(a) - depth(b));
    this.restQ = new Map(this.driven.map((o) => [o, o.getWorldQuaternion(new T.Quaternion())]));
    this.restP = new Map(this.driven.map((o) => [o, P(o)]));
    this.restLocal = new Map(this.driven.map((o) => [o, { p: o.position.clone(), q: o.quaternion.clone() }]));
    // helper bones placed on a driven joint but parented elsewhere (e.g. the horse's hip bone hangs off the second spine
    // bone while it sits on the pelvis): they follow the joint they sit on, else the skin there tears when the back bends
    const skinned = new Set(); gltf.scene.traverse((o) => { if (o.isSkinnedMesh) o.skeleton.bones.forEach((b) => skinned.add(b)); });
    const tol = 0.02 * cfg.hips; this.followers = [];
    for (const b of skinned) {
      if (this.restP.has(b)) continue;
      const pb = P(b); let best = null, bd = Infinity;
      for (const d of this.driven) { const dd = this.restP.get(d).distanceTo(pb); if (dd < bd) { bd = dd; best = d; } }
      if (best && bd < tol && best !== b.parent && !this.driven.some((d) => this.isAncestor(b, d))) { // never a bone above a driven one
        const off = this.restQ.get(best).clone().invert().multiply(b.getWorldQuaternion(new T.Quaternion()));
        this.followers.push({ b, to: best, off });
      }
    }
    for (const l of [this.front.L, this.front.R, this.hind.L, this.hind.R]) {
      l.tip0 = tipP(l); const e = l.bones[l.bones.length - 1]; l.metaLen = e ? l.tip0.distanceTo(this.restP.get(e)) : 0;
      // toe bone between the paw bone and the tip: kept at its standing angle while the paw is planted
      const end = e || l.bones[l.bones.length - 2]; let t = l.tip[0];
      while (t && t.parent && t.parent !== end) t = t.parent;
      if (t && t.parent === end && t !== l.tip[0]) {
        l.toe = t; l.toeP0 = P(t); l.toeQ0 = t.getWorldQuaternion(new T.Quaternion());
        const d = l.tip0.clone().sub(l.toeP0).normalize(); l.toeElev0 = Math.asin(Math.max(-1, Math.min(1, d.y)));
      }
    }
    const fwd0 = P(this.head).sub(P(this.pelvis)).setY(0).normalize();
    this.nose0 = this.noseB.length ? this.noseB.reduce((a, o) => a.add(P(o)), new T.Vector3()).multiplyScalar(1 / this.noseB.length) : this.autoNose(gltf.scene, this.head, fwd0);
    // torso chain pelvis -> spine -> first neck bone, and the point matching the network's Spine1 (0.19 of 0.415)
    const chain = [this.pelvis, ...this.spine, this.neck[0]].map((o) => this.restP.get(o));
    const cum = [0]; for (let i = 1; i < chain.length; i++) cum.push(cum[i - 1] + chain[i].distanceTo(chain[i - 1]));
    this.torsoLen = cum[cum.length - 1];
    const at = (u) => { const d = u * this.torsoLen; let i = 1; while (i < cum.length - 1 && cum[i] < d) i++; return chain[i - 1].clone().lerp(chain[i], (d - cum[i - 1]) / Math.max(1e-6, cum[i] - cum[i - 1])); };
    this.spine1 = at(0.19 / 0.415);
    this.spineU = this.spine.map((o) => { const p = this.restP.get(o); let best = 0, bd = Infinity; for (let k = 0; k <= 40; k++) { const d = at(k / 40).distanceTo(p); if (d < bd) { bd = d; best = k / 40; } } return best; });
    const lr = (a, b) => this.restP.get(a).clone().sub(this.restP.get(b)).normalize();
    this.hindL0 = lr(this.hind.L.bones[0], this.hind.R.bones[0]);
    this.frontL0 = lr(this.front.L.bones[1], this.front.R.bones[1]);
    { const x = this.restP.get(this.neck[0]).clone().sub(this.spine1).normalize(), z = this.frontL0.clone().addScaledVector(x, -this.frontL0.dot(x)).normalize();
      this.qCm0inv = new T.Quaternion().setFromRotationMatrix(new T.Matrix4().makeBasis(x, z.clone().cross(x), z)).invert(); }
    // paw pad: how far the toe-tip joints sit above the lowest point of the mesh
    let lowest = Infinity; gltf.scene.traverse((o) => { if (o.isSkinnedMesh) { o.computeBoundingBox(); lowest = Math.min(lowest, o.boundingBox.clone().applyMatrix4(o.matrixWorld).min.y); } });
    const tipY = Math.min(...[this.front.L, this.front.R, this.hind.L, this.hind.R].map((l) => l.tip0.y));
    this.pawR = Math.max(0, Math.min(0.06 * cfg.hips, tipY - (Number.isFinite(lowest) ? lowest : tipY)));
    this.tmp = { q: new T.Quaternion(), q2: new T.Quaternion(), m: new T.Matrix4(), m2: new T.Matrix4(), v: new T.Vector3(), v2: new T.Vector3() };
    this.body = new AnimalBodyFX(T, gltf.scene, this);
    // soles: how far below each toe-tip joint the paw's skin reaches in the flat reference pose (so a planted paw's
    // sole, not its tip joint, meets the ground)
    const vtx = new T.Vector3();
    for (const l of [this.front.L, this.front.R, this.hind.L, this.hind.R]) {
      const bones = l.bones.filter(Boolean), under = new Set(); bones[bones.length - 1].traverse((o) => under.add(o));
      let lo = Infinity; const cand = [];
      gltf.scene.traverse((o) => { if (!o.isSkinnedMesh) return; const si = o.geometry.attributes.skinIndex, sw = o.geometry.attributes.skinWeight, n = o.geometry.attributes.position.count;
        for (let i = 0; i < n; i++) { let w = 0; for (let k = 0; k < 4; k++) if (under.has(o.skeleton.bones[si.getComponent(i, k)])) w += sw.getComponent(i, k);
          if (w > 0.6) { o.getVertexPosition(i, vtx).applyMatrix4(o.matrixWorld); cand.push([vtx.y, o, i]); if (vtx.y < lo) lo = vtx.y; } } });
      // the sole can sit above the tip joint (the fox's toe-end bones reach below its paws)
      l.sole = Number.isFinite(lo) ? Math.max(-0.15 * cfg.hips, Math.min(0.15 * cfg.hips, l.tip0.y - lo)) : 0;
      // the lowest skin points of the paw in the reference pose, skinned each frame to measure the real sole height
      cand.sort((a, b) => a[0] - b[0]);
      l.soleV = cand.slice(0, 48).map(([, o, i]) => { const si = o.geometry.attributes.skinIndex, sw = o.geometry.attributes.skinWeight;
        return { o, v: new T.Vector3().fromBufferAttribute(o.geometry.attributes.position, i), b: [0, 1, 2, 3].map((k) => si.getComponent(i, k)), w: [0, 1, 2, 3].map((k) => sw.getComponent(i, k)) }; });
      l.under = bones[bones.length - 1];
    }
    this.netPlantY = [null, null, null, null]; this.soleCorr = [0, 0, 0, 0]; this.drop = 0; this.look = {}; this.skinM = new T.Matrix4();
  }
  // leg chain lengths (metres) from the shoulder / hip joint to the toe tip, averaged over both sides; the network's
  // legs are scaled uniformly to these, so its paws are within reach of the model's legs
  legLengths() {
    const P = (o) => this.restP.get(o), len = (l, from) => { const pts = [...l.bones.slice(from).filter(Boolean).map(P), l.tip0]; let d = 0; for (let i = 1; i < pts.length; i++) d += pts[i].distanceTo(pts[i - 1]); return d; };
    return { front: (len(this.front.L, 1) + len(this.front.R, 1)) / 2, hind: (len(this.hind.L, 0) + len(this.hind.R, 0)) / 2 };
  }
  // R such that R * (rest aim, rest left) = (aim, left): rigid rotation of a body part from its rest pose
  align(a0, l0, a1, l1, out) {
    const T = this.T, f = (a, l, m) => { const x = a.clone().normalize(); let z = l.clone().addScaledVector(x, -l.dot(x)); if (z.lengthSq() < 1e-8) z = new T.Vector3(0, 1, 0).cross(x); z.normalize(); const y = z.clone().cross(x); return m.makeBasis(x, y, z); };
    const F0 = f(a0, l0, new T.Matrix4()), F1 = f(a1, l1, new T.Matrix4());
    return out.setFromRotationMatrix(F1.multiply(F0.transpose()));
  }
  isAncestor(a, b) { for (let x = b.parent; x; x = x.parent) if (x === a) return true; return false; }
  setWorldQ(o, q) { const pq = this.tmp.q2; o.parent.getWorldQuaternion(pq); o.quaternion.copy(pq.invert().multiply(q)); o.updateWorldMatrix(false, false); }
  // W: network joint positions (world, model proportions), ix: name -> index, rest: { pelvis frame and tail direction at rest }
  apply(W, ix, rest, contacts, dt = 1 / 60) {
    const T = this.T, V = (n) => { const i = ix[n] * 3; return new T.Vector3(W[i], W[i + 1], W[i + 2]); };
    const hips = V('Hips'), sp1 = V('Spine1'), neck = V('Neck'), head = V('Head'), headEnd = V('HeadEnd');
    const hindL = V('LeftUpLeg').sub(V('RightUpLeg')).normalize(), frontL = V('LeftShoulder').sub(V('RightShoulder')).normalize();
    const R = new Map(), rot = (a0, l0, a1, l1) => this.align(a0, l0, a1, l1, new T.Quaternion());
    const Rp = rot(this.spine1.clone().sub(this.restP.get(this.pelvis)), this.hindL0, sp1.clone().sub(hips), hindL);
    const Rc = rot(this.restP.get(this.neck[0]).clone().sub(this.spine1), this.frontL0, neck.clone().sub(sp1), frontL);
    R.set(this.root, Rp); R.set(this.pelvis, Rp);
    this.spine.forEach((o, i) => { const u = this.spineU[i]; R.set(o, Rp.clone().slerp(Rc, u * u * (3 - 2 * u))); });
    if (this.cfg.carriage !== 'model' || !rest) { // neck and head along the network's (dog's) carriage
      const lift = (this.cfg.headLift || 0) * Math.PI / 180, up = (v, k) => (lift ? v.applyAxisAngle(frontL, -lift * k) : v); // species that carry the head higher than a dog
      const Rn = rot(this.restP.get(this.head).clone().sub(this.restP.get(this.neck[0])), this.frontL0, up(head.clone().sub(neck), 1), frontL);
      this.neck.forEach((o) => R.set(o, Rn));
      R.set(this.head, rot(this.nose0.clone().sub(this.restP.get(this.head)), this.frontL0, up(headEnd.clone().sub(head), 0.6), frontL));
    } else { // the species' own carriage, moved by how the network's neck and head turn relative to its chest
      const fr = (a, l) => { const x = a.clone().normalize(), z = l.clone().addScaledVector(x, -l.dot(x)).normalize(); return new T.Matrix4().makeBasis(x, z.clone().cross(x), z); };
      const Cn = fr(neck.clone().sub(sp1), frontL), CnT = Cn.clone().transpose(), qCn = new T.Quaternion().setFromRotationMatrix(Cn);
      const local = (d) => d.normalize().applyMatrix4(CnT);
      const dN = new T.Quaternion().setFromUnitVectors(rest.neckLocal, local(head.clone().sub(neck))), dH = new T.Quaternion().setFromUnitVectors(rest.headLocal, local(headEnd.clone().sub(head)));
      const lim = (this.cfg.carriageLimit || 180) * Math.PI / 180, clampQ = (q) => { const a = 2 * Math.acos(Math.min(1, Math.abs(q.w))); return a > lim ? new T.Quaternion().slerp(q, lim / a) : q; };
      dN.copy(clampQ(dN)); dH.copy(clampQ(dH));
      const toW = (d) => qCn.clone().multiply(d).multiply(this.qCm0inv);
      const Rn = toW(dN); this.neck.forEach((o) => R.set(o, Rn)); R.set(this.head, toW(dH));
    }
    // tail: the network's tail swing relative to its pelvis, applied on top of the model's own tail shape
    if (this.tail.length && rest) {
      const fr = (a, l) => { const x = a.clone().normalize(), z = l.clone().addScaledVector(x, -l.dot(x)).normalize(); return new T.Matrix4().makeBasis(x, z.clone().cross(x), z); };
      const Pn = fr(sp1.clone().sub(hips), hindL), d1 = V('Tail1End').sub(V('Tail')).normalize();
      const dl = d1.clone().applyMatrix4(Pn.clone().transpose()), delta = new T.Quaternion().setFromUnitVectors(rest.tailLocal, dl);
      const qPn = new T.Quaternion().setFromRotationMatrix(Pn), Rt = qPn.clone().multiply(delta).multiply(qPn.clone().invert()).multiply(Rp);
      this.tail.forEach((o, i) => R.set(o, Rp.clone().slerp(Rt, (i + 1) / this.tail.length)));
    }
    // limbs: each bone along its network segment
    const limbs = [['front', 'L', 'Left'], ['front', 'R', 'Right'], ['hind', 'L', 'Left'], ['hind', 'R', 'Right']];
    for (const [kind, s, S] of limbs) {
      const l = this[kind][s], P = (o) => this.restP.get(o), left = kind === 'front' ? frontL : hindL, left0 = kind === 'front' ? this.frontL0 : this.hindL0;
      const pts = kind === 'front' ? [`${S}Shoulder`, `${S}Arm`, `${S}ForeArm`, `${S}Hand`, `${S}HandEnd`] : [null, `${S}UpLeg`, `${S}Leg`, `${S}Foot`, `${S}FootEnd`];
      const bones = kind === 'front' ? l.bones : [null, ...l.bones];
      for (let k = 0; k < 4; k++) {
        const o = bones[k]; if (!o) continue;
        let nx = k + 1; while (nx < 4 && !bones[nx]) nx++;
        const restTo = nx < 4 ? P(bones[nx]) : l.tip0, netFrom = pts[k] ?? pts[k + 1], netTo = pts[nx] ?? pts[4];
        R.set(o, rot(restTo.clone().sub(P(o)), left0, V(netTo).sub(V(netFrom)), left));
      }
    }
    // world rotations (parents first), then the root's translation so that the pelvis lands on the network's hips
    // head and neck: limited relative to the chest and smoothed (the dog data looks around fast; on other bodies the
    // head spun up to 500 deg/s and turned 50-90 deg sideways)
    const Rn0 = R.get(this.neck[0]), Rh0 = R.get(this.head);
    if (Rn0) { const Rn1 = this.limitLook(Rn0, Rc, NECK_LIMIT, 'neck', dt); this.neck.forEach((o) => R.set(o, Rn1)); }
    if (Rh0) R.set(this.head, this.limitLook(Rh0, Rc, HEAD_LIMIT, 'head', dt));
    for (const o of this.driven) { const q = R.get(o); if (q) this.setWorldQ(o, q.clone().multiply(this.restQ.get(o))); }
    this.limitTwist(this.head, HEAD_ROLL_MAX, null, this.nose0.clone().sub(this.restP.get(this.head)));
    const hipNet = V('LeftUpLeg').add(V('RightUpLeg')).multiplyScalar(0.5), hipModel = this.hind.L.bones[0].getWorldPosition(new T.Vector3()).add(this.hind.R.bones[0].getWorldPosition(new T.Vector3())).multiplyScalar(0.5);
    const rootTo = this.root.getWorldPosition(new T.Vector3()).add(hipNet.sub(hipModel)); rootTo.y -= this.drop; // legs that cannot reach the ground lower the body
    this.root.position.copy(this.root.parent.worldToLocal(rootTo)); this.root.updateMatrixWorld(true);
    // Paws: a planted paw stands flat in its bind orientation, turned to the body's heading; a lifted paw flexes as the
    // network's does (wrist -> toe tip direction), at most PAW_FLEX from flat and without twist. The two-bone IK then
    // puts the wrist / ankle where that paw reaches the network's toe tip. (Aiming the paw bone at the tip with a
    // minimal rotation bent paws 80-140 deg and twisted them 20-60 deg, because rigs and network split paws differently.)
    const yawOf = (Rq) => { const f = new T.Vector3(0, 0, 1).applyQuaternion(Rq); return new T.Quaternion().setFromAxisAngle(new T.Vector3(0, 1, 0), Math.atan2(f.x, f.z)); };
    const yawF = yawOf(Rc), yawH = yawOf(Rp); let need = 0, slack = Infinity; const planted = [];
    for (const [kind, s, S] of limbs) {
      const l = this[kind][s], b = kind === 'front' ? l.bones.slice(1) : l.bones, [u, m, e] = b;
      const tipT = V(kind === 'front' ? `${S}HandEnd` : `${S}FootEnd`), wristN = V(kind === 'front' ? `${S}Hand` : `${S}Foot`);
      const li = limbs.findIndex((x) => x[0] === kind && x[1] === s), c = contacts ? contacts[li] : 0, w = Math.max(0, Math.min(1, (c - 0.3) / 0.4));
      // the network plants its toe tip at its own height; shift so this model's sole touches the ground when planted
      if (w > 0.9) this.netPlantY[li] = this.netPlantY[li] === null ? tipT.y : this.netPlantY[li] + (tipT.y - this.netPlantY[li]) * Math.min(1, dt / 0.25);
      const plantY = this.netPlantY[li] ?? tipT.y;
      tipT.y += l.sole - plantY + w * (plantY - tipT.y) + this.soleCorr[li]; // planted: sole on the ground; lifted: the network's lift
      let target = tipT, Qpaw = null;
      if (e) {
        const metaDir = tipT.clone().sub(wristN).normalize(), stiff = this.cfg.style?.metaStiff || 0;
        if (stiff > 0) { const down = new T.Vector3(0, -1, 0).applyQuaternion(kind === 'front' ? Rc : Rp); metaDir.lerp(down, stiff).normalize(); } // standing cannon: along the body's down axis
        const restQ = this.restQ.get(e), restOff = l.tip0.clone().sub(this.restP.get(e)), yq = kind === 'front' ? yawF : yawH;
        const Qplant = yq.clone().multiply(restQ);
        const Qswing = new T.Quaternion().setFromUnitVectors(restOff.clone().applyQuaternion(yq).normalize(), metaDir).multiply(Qplant);
        const ang = Qplant.angleTo(Qswing); if (ang > PAW_FLEX) Qswing.copy(Qplant.clone().slerp(Qswing, PAW_FLEX / ang));
        Qpaw = Qswing.clone().slerp(Qplant, w * (kind === 'front' ? 1 : HIND_PLANT));
        target = tipT.clone().sub(restOff.clone().applyQuaternion(restQ.clone().invert()).applyQuaternion(Qpaw));
      }
      const A = u.getWorldPosition(new T.Vector3()), Kn = m.getWorldPosition(new T.Vector3());
      const C = e ? e.getWorldPosition(new T.Vector3()) : this.tipNow(l, m);
      this.twoBone(u, m, A, Kn, C, target);
      if (Qpaw) this.setWorldQ(e, Qpaw);
      if (w > 0.5) { const l1 = Kn.distanceTo(A), l2 = C.distanceTo(Kn), d = target.distanceTo(A) - 0.98 * (l1 + l2); need = Math.max(need, d); slack = Math.min(slack, -d); }
      planted.push([l, li, w]);
      // no candy-wrapper: limb bones twist at most TWIST_MAX about their own axis (children keep their world rotation)
      this.limitTwist(u, TWIST_MAX, m); this.limitTwist(m, TWIST_MAX, e || null);
      // planted paw: toes back at their standing angle (the paw presses flat instead of pivoting as one block)
      if (l.toe && w > 0) {
        const t = l.toe, tq = t.getWorldQuaternion(new T.Quaternion()), tp = t.getWorldPosition(new T.Vector3());
        const cur = l.tip0.clone().sub(l.toeP0).applyQuaternion(l.toeQ0.clone().invert()).applyQuaternion(tq).normalize();
        const h = new T.Vector3(cur.x, 0, cur.z); if (h.lengthSq() < 1e-6) continue; h.normalize();
        const want = h.multiplyScalar(Math.cos(l.toeElev0)).add(new T.Vector3(0, Math.sin(l.toeElev0), 0));
        const dir = cur.clone().lerp(want, w).normalize();
        this.setWorldQ(t, new T.Quaternion().setFromUnitVectors(cur, dir).multiply(tq)); void tp;
      }
    }
    this.applyFollowers();
    // planted paws the legs cannot reach lower the body; slack in every planted leg lets it rise again (next frame)
    const dropT = Math.max(0, Math.min(0.12 * this.hips, this.drop + (need > 0 ? need : Number.isFinite(slack) ? -slack : -this.drop)));
    this.drop += (dropT - this.drop) * Math.min(1, dt / 0.15);
    // real sole height of each planted paw (its lowest skin points): integrate it out of the paw's target
    for (const [l, li, w] of planted) {
      if (!l.soleV.length) continue;
      l.under.updateMatrixWorld(true);
      let lo = Infinity; const v = new this.T.Vector3();
      for (const sv of l.soleV) {
        const out = new this.T.Vector3(), bones = sv.o.skeleton.bones, inv = sv.o.skeleton.boneInverses;
        for (let k = 0; k < 4; k++) { if (sv.w[k] <= 0) continue; this.skinM.multiplyMatrices(bones[sv.b[k]].matrixWorld, inv[sv.b[k]]).multiply(sv.o.bindMatrix); out.addScaledVector(v.copy(sv.v).applyMatrix4(this.skinM), sv.w[k]); }
        if (out.y < lo) lo = out.y;
      }
      this.soleCorr[li] = Math.max(-0.15 * this.hips, Math.min(0.15 * this.hips, this.soleCorr[li] - lo * w * Math.min(1, dt / 0.08)));
    }
  }
  // rotation Rw (world, from rest) of the neck or head limited relative to the chest rotation Rc: yaw, pitch (positive =
  // nose down) and roll in the upright rest frame, then smoothed toward the limited target
  limitLook(Rw, Rc, lim, key, dt) {
    const T = this.T, rel = Rc.clone().invert().multiply(Rw), e = new T.Euler().setFromQuaternion(rel, 'YXZ'), r = Math.PI / 180;
    e.y = Math.max(-lim.yaw * r, Math.min(lim.yaw * r, e.y)); e.x = Math.max(-lim.up * r, Math.min(lim.down * r, e.x)); e.z = Math.max(-lim.roll * r, Math.min(lim.roll * r, e.z));
    const tgt = new T.Quaternion().setFromEuler(e), prev = this.look[key];
    const q = prev ? prev.clone().slerp(tgt, 1 - Math.exp(-dt / LOOK_SMOOTH)) : tgt;
    this.look[key] = q;
    return Rc.clone().multiply(q);
  }
  // helpers that sit on a driven joint: same world rotation change as that joint, kept on its position
  applyFollowers() {
    const T = this.T;
    for (const f of this.followers) {
      const q = f.to.getWorldQuaternion(new T.Quaternion()).multiply(f.off);
      this.setWorldQ(f.b, q);
      const pw = f.to.getWorldPosition(new T.Vector3()); f.b.position.copy(f.b.parent.worldToLocal(pw)); f.b.updateWorldMatrix(false, true);
    }
  }
  // clamp a bone's twist about its own axis (toward its child, or restAxisWorld) relative to the reference pose; the
  // given (IK-placed) child keeps its world rotation, every other child follows
  limitTwist(o, max, child = null, restAxisWorld = null) {
    const T = this.T, r = this.restLocal.get(o); if (!r) return;
    let axis;
    if (restAxisWorld) axis = restAxisWorld.clone().applyQuaternion(this.restQ.get(o).clone().invert()).normalize(); // into the bone's rest frame
    else { const ch = child && child.parent === o ? child : o.children.find((x) => this.restLocal.has(x)) || o.children.find((x) => x.isBone); if (!ch) return;
      axis = (this.restLocal.get(ch)?.p || ch.position).clone().applyQuaternion(r.q.clone().invert().multiply(r.q)).normalize(); }
    if (!(axis.lengthSq() > 0.5)) return;
    const D = r.q.clone().invert().multiply(o.quaternion), pr = D.x * axis.x + D.y * axis.y + D.z * axis.z;
    const tw = new T.Quaternion(axis.x * pr, axis.y * pr, axis.z * pr, D.w).normalize(); if (tw.w < 0) tw.set(-tw.x, -tw.y, -tw.z, -tw.w);
    const ang = 2 * Math.acos(Math.min(1, tw.w)); if (ang <= max) return;
    const swing = D.clone().multiply(tw.clone().invert()), keep = tw.clone().slerp(new T.Quaternion(), 1 - max / ang);
    const kids = child && child.parent === o ? [[child, child.getWorldQuaternion(new T.Quaternion())]] : []; // only the IK-placed child stays put; helper and face bones follow
    o.quaternion.copy(r.q.clone().multiply(swing).multiply(keep)); o.updateWorldMatrix(false, false);
    for (const [x, q] of kids) this.setWorldQ(x, q);
  }
  // current toe tip, carried rigidly by bone o from its rest pose
  tipNow(l, o) {
    const T = this.T, wq = o.getWorldQuaternion(new T.Quaternion()), rq = this.restQ.get(o).clone().invert();
    return l.tip0.clone().sub(this.restP.get(o)).applyQuaternion(rq).applyQuaternion(wq).add(o.getWorldPosition(new T.Vector3()));
  }
  twoBone(u, m, A, Kn, C, target) {
    const T = this.T, l1 = Kn.distanceTo(A), l2 = C.distanceTo(Kn), d = target.clone().sub(A);
    const dist = Math.min(Math.max(d.length(), Math.abs(l1 - l2) + 1e-4), l1 + l2 - 1e-4), dn = d.normalize();
    const a = (l1 * l1 - l2 * l2 + dist * dist) / (2 * dist), h = Math.sqrt(Math.max(0, l1 * l1 - a * a));
    const pole = Kn.clone().sub(A), pp = pole.addScaledVector(dn, -pole.dot(dn)); if (pp.lengthSq() < 1e-10) return;
    const K2 = A.clone().addScaledVector(dn, a).addScaledVector(pp.normalize(), h);
    const q1 = new T.Quaternion().setFromUnitVectors(Kn.clone().sub(A).normalize(), K2.clone().sub(A).normalize());
    this.setWorldQ(u, q1.multiply(u.getWorldQuaternion(new T.Quaternion())));
    const Kw = m.getWorldPosition(new T.Vector3()), Cw = C.clone().sub(A).applyQuaternion(new T.Quaternion().setFromUnitVectors(Kn.clone().sub(A).normalize(), K2.clone().sub(A).normalize())).add(A);
    const q2 = new T.Quaternion().setFromUnitVectors(Cw.sub(Kw).normalize(), A.clone().addScaledVector(dn, dist).sub(Kw).normalize());
    this.setWorldQ(m, q2.multiply(m.getWorldQuaternion(new T.Quaternion())));
  }
  // snout tip of rigs without a nose bone: the frontmost rest vertex that mostly follows the head
  autoNose(scene, head, fwd) {
    const T = this.T, v = new T.Vector3(); let best = null, bd = -Infinity;
    const comp = (a, i, k) => (k === 0 ? a.getX(i) : k === 1 ? a.getY(i) : k === 2 ? a.getZ(i) : a.getW(i));
    scene.traverse((o) => {
      if (!o.isSkinnedMesh) return;
      const under = o.skeleton.bones.map((b) => { for (let p = b; p; p = p.parent) if (p === head) return true; return false; });
      const g = o.geometry, si = g.attributes.skinIndex, sw = g.attributes.skinWeight;
      for (let i = 0; i < g.attributes.position.count; i++) {
        let w = 0; for (let k = 0; k < 4; k++) if (under[comp(si, i, k)]) w += comp(sw, i, k);
        if (w < 0.5) continue;
        o.getVertexPosition(i, v); v.applyMatrix4(o.matrixWorld);
        const d = v.dot(fwd); if (d > bd) { bd = d; best = v.clone(); }
      }
    });
    return best || head.getWorldPosition(new T.Vector3()).addScaledVector(fwd, 0.15 * this.hips);
  }
  setVisible(v) { this.group.visible = v; }
}
