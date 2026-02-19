package fieldtrie

import (
	"reflect"
	"sync"

	"github.com/OffchainLabs/prysm/v7/beacon-chain/state/state-native/types"
	"github.com/OffchainLabs/prysm/v7/beacon-chain/state/stateutil"
	multi_value_slice "github.com/OffchainLabs/prysm/v7/container/multi-value-slice"
	"github.com/OffchainLabs/prysm/v7/container/trie"
	"github.com/OffchainLabs/prysm/v7/crypto/hash"
	pmath "github.com/OffchainLabs/prysm/v7/math"
	"github.com/pkg/errors"
)

var (
	ErrInvalidFieldTrie = errors.New("invalid field trie")
	ErrEmptyFieldTrie   = errors.New("empty field trie")
)

// sliceAccessor describes an interface for a multivalue slice
// object that returns information about the multivalue slice along with the
// particular state instance we are referencing.
type sliceAccessor interface {
	Len(obj multi_value_slice.Identifiable) int
	State() multi_value_slice.Identifiable
}

// OverlayPromotionThreshold is the maximum number of overlay entries
// before an overlay is promoted to an owned trie. Epoch boundaries
// typically dirty all ~1.1M validators at once, so this threshold
// catches that case before adding entries to override maps.
// Exported so that callers (e.g. recomputeFieldTrie) can detect large
// dirty sets and choose a from-scratch rebuild instead.
const OverlayPromotionThreshold = 10000

// FieldTrie is the representation of the representative
// trie of the particular field.
//
// A FieldTrie operates in one of two modes:
//   - Owned mode: fieldLayers != nil, base == nil. The trie owns its full
//     layer data and mutations happen in-place.
//   - Overlay mode: fieldLayers == nil, base != nil. The trie stores only
//     sparse diffs (overrides) against an immutable base trie. Root computation
//     walks the base read-only, substituting override values at modified positions.
type FieldTrie struct {
	*sync.RWMutex
	reference     *stateutil.Reference
	fieldLayers   [][]*[32]byte         // non-nil in owned mode
	base          *FieldTrie            // non-nil in overlay mode; immutable, ref-counted
	overrides     []map[uint64][32]byte // per-level sparse modifications [level][nodeIdx] → hash
	field         types.FieldIndex
	dataType      types.DataType
	length        uint64
	numOfElems    int
	isTransferred bool
}

// NewFieldTrie is the constructor for the field trie data structure. It creates the corresponding
// trie according to the given parameters. Depending on whether the field is a basic/composite array
// which is either fixed/variable length, it will appropriately determine the trie.
func NewFieldTrie(field types.FieldIndex, fieldInfo types.DataType, elements any, length uint64) (*FieldTrie, error) {
	if elements == nil {
		return &FieldTrie{
			field:      field,
			dataType:   fieldInfo,
			reference:  stateutil.NewRef(1),
			RWMutex:    new(sync.RWMutex),
			length:     length,
			numOfElems: 0,
		}, nil
	}

	fieldRoots, err := fieldConverters(field, []uint64{}, elements, true)
	if err != nil {
		return nil, err
	}

	if err := validateElements(field, fieldInfo, elements, length); err != nil {
		return nil, err
	}
	var numOfElems int
	if val, ok := elements.(sliceAccessor); ok {
		numOfElems = val.Len(val.State())
	} else {
		numOfElems = reflect.Indirect(reflect.ValueOf(elements)).Len()
	}
	switch fieldInfo {
	case types.BasicArray:
		fl, err := stateutil.ReturnTrieLayer(fieldRoots, length)
		if err != nil {
			return nil, err
		}
		return &FieldTrie{
			fieldLayers: fl,
			field:       field,
			dataType:    fieldInfo,
			reference:   stateutil.NewRef(1),
			RWMutex:     new(sync.RWMutex),
			length:      length,
			numOfElems:  numOfElems,
		}, nil
	case types.CompositeArray, types.CompressedArray:
		return &FieldTrie{
			fieldLayers: stateutil.ReturnTrieLayerVariable(fieldRoots, length),
			field:       field,
			dataType:    fieldInfo,
			reference:   stateutil.NewRef(1),
			RWMutex:     new(sync.RWMutex),
			length:      length,
			numOfElems:  numOfElems,
		}, nil
	default:
		return nil, errors.Errorf("unrecognized data type in field map: %v", reflect.TypeFor[types.DataType]().Name())
	}
}

// RecomputeTrie rebuilds the affected branches in the trie according to the provided
// changed indices and elements. This recomputes the trie according to the particular
// field the trie is based on.
func (f *FieldTrie) RecomputeTrie(indices []uint64, elements any) ([32]byte, error) {
	f.Lock()
	defer f.Unlock()
	if len(indices) == 0 {
		return f.TrieRoot()
	}

	fieldRoots, err := fieldConverters(f.field, indices, elements, false)
	if err != nil {
		return [32]byte{}, err
	}

	if err := f.validateIndices(indices); err != nil {
		return [32]byte{}, err
	}
	if val, ok := elements.(sliceAccessor); ok {
		f.numOfElems = val.Len(val.State())
	} else {
		f.numOfElems = reflect.Indirect(reflect.ValueOf(elements)).Len()
	}

	// Dispatch to overlay or owned recomputation.
	if f.base != nil {
		return f.recomputeOverlayDispatch(indices, fieldRoots)
	}
	return f.recomputeOwned(indices, fieldRoots)
}

// CopyTrie creates a lightweight overlay copy of the trie. Instead of
// deep-copying all layer data (O(N) pointers), the copy shares the
// immutable base and stores only sparse diffs. This is O(1) for a
// fresh copy from an owned trie, or O(K) where K is the number of
// existing overrides when copying from another overlay.
func (f *FieldTrie) CopyTrie() *FieldTrie {
	if f.Empty() {
		return &FieldTrie{
			field:      f.field,
			dataType:   f.dataType,
			reference:  stateutil.NewRef(1),
			RWMutex:    new(sync.RWMutex),
			length:     f.length,
			numOfElems: f.numOfElems,
		}
	}
	if f.base != nil {
		// Source is an overlay: new overlay sharing the same base.
		f.base.reference.AddRef()
		return &FieldTrie{
			base:       f.base,
			overrides:  copyOverrides(f.overrides),
			field:      f.field,
			dataType:   f.dataType,
			reference:  stateutil.NewRef(1),
			RWMutex:    new(sync.RWMutex),
			length:     f.length,
			numOfElems: f.numOfElems,
		}
	}
	// Source is owned: create an overlay on this trie.
	f.reference.AddRef()
	depth := len(f.fieldLayers)
	overrides := make([]map[uint64][32]byte, depth)
	return &FieldTrie{
		base:       f,
		overrides:  overrides,
		field:      f.field,
		dataType:   f.dataType,
		reference:  stateutil.NewRef(1),
		RWMutex:    new(sync.RWMutex),
		length:     f.length,
		numOfElems: f.numOfElems,
	}
}

// Length return the length of the whole field trie.
func (f *FieldTrie) Length() uint64 {
	return f.length
}

// TransferTrie starts the process of transferring all the
// trie related data to a new trie. This is done if we
// know that other states which hold references to this
// trie will unlikely need it for recomputation. This helps
// us save on a copy. Any caller of this method will need
// to take care that this isn't called on an empty trie.
func (f *FieldTrie) TransferTrie() *FieldTrie {
	// Overlays cannot be transferred (the base is immutable and shared).
	// Just create a cheap overlay copy instead.
	if f.base != nil {
		return f.CopyTrie()
	}
	if f.fieldLayers == nil {
		return &FieldTrie{
			field:      f.field,
			dataType:   f.dataType,
			reference:  stateutil.NewRef(1),
			RWMutex:    new(sync.RWMutex),
			length:     f.length,
			numOfElems: f.numOfElems,
		}
	}
	f.isTransferred = true
	nTrie := &FieldTrie{
		fieldLayers: f.fieldLayers,
		field:       f.field,
		dataType:    f.dataType,
		reference:   stateutil.NewRef(1),
		RWMutex:     new(sync.RWMutex),
		length:      f.length,
		numOfElems:  f.numOfElems,
	}
	// Zero out field layers here.
	f.fieldLayers = nil
	return nTrie
}

// TrieRoot returns the corresponding root of the trie.
func (f *FieldTrie) TrieRoot() ([32]byte, error) {
	if f.Empty() {
		return [32]byte{}, ErrEmptyFieldTrie
	}

	// Overlay mode: read root from overrides, fallback to base.
	if f.base != nil {
		depth := len(f.base.fieldLayers) - 1
		trieRoot := f.readOverlayNode(depth, 0)
		switch f.dataType {
		case types.BasicArray:
			return trieRoot, nil
		case types.CompositeArray:
			leafCount := uint64(len(f.base.fieldLayers[0]))
			for idx := range f.overrides[0] {
				if idx+1 > leafCount {
					leafCount = idx + 1
				}
			}
			return stateutil.AddInMixin(trieRoot, leafCount)
		case types.CompressedArray:
			return stateutil.AddInMixin(trieRoot, uint64(f.numOfElems))
		default:
			return [32]byte{}, errors.Errorf("unrecognized data type in field map: %v", reflect.TypeFor[types.DataType]().Name())
		}
	}

	// Owned mode.
	if len(f.fieldLayers[len(f.fieldLayers)-1]) == 0 {
		return [32]byte{}, ErrInvalidFieldTrie
	}
	switch f.dataType {
	case types.BasicArray:
		return *f.fieldLayers[len(f.fieldLayers)-1][0], nil
	case types.CompositeArray:
		trieRoot := *f.fieldLayers[len(f.fieldLayers)-1][0]
		return stateutil.AddInMixin(trieRoot, uint64(len(f.fieldLayers[0])))
	case types.CompressedArray:
		trieRoot := *f.fieldLayers[len(f.fieldLayers)-1][0]
		return stateutil.AddInMixin(trieRoot, uint64(f.numOfElems))
	default:
		return [32]byte{}, errors.Errorf("unrecognized data type in field map: %v", reflect.TypeFor[types.DataType]().Name())
	}
}

// FieldReference returns the underlying field reference
// object for the trie.
func (f *FieldTrie) FieldReference() *stateutil.Reference {
	return f.reference
}

// Empty checks whether the underlying field trie is
// empty or not.
func (f *FieldTrie) Empty() bool {
	return f == nil || (f.fieldLayers == nil && f.base == nil) || f.isTransferred
}

// IsOverlay returns true if this trie operates in overlay mode,
// storing sparse diffs against an immutable base.
func (f *FieldTrie) IsOverlay() bool {
	return f != nil && f.base != nil
}

// ReleaseBase decrements the base trie's reference count and clears
// the overlay's reference to it. Called during finalizer cleanup to
// allow the base to be garbage collected when no overlays reference it.
func (f *FieldTrie) ReleaseBase() {
	if f.base != nil {
		f.base.reference.MinusRef()
		f.base = nil
		f.overrides = nil
	}
}

// overlaySize returns the total number of entries across all override maps.
func (f *FieldTrie) overlaySize() int {
	n := 0
	for _, m := range f.overrides {
		n += len(m)
	}
	return n
}

// copyOverrides returns a deep copy of the override maps.
func copyOverrides(src []map[uint64][32]byte) []map[uint64][32]byte {
	if src == nil {
		return nil
	}
	dst := make([]map[uint64][32]byte, len(src))
	for i, m := range src {
		if len(m) > 0 {
			dst[i] = make(map[uint64][32]byte, len(m))
			for k, v := range m {
				dst[i][k] = v
			}
		}
	}
	return dst
}

// readOverlayNode reads a node from the overlay at (level, idx).
// Priority: overrides → base.fieldLayers → trie.ZeroHashes.
func (f *FieldTrie) readOverlayNode(level int, idx uint64) [32]byte {
	if m := f.overrides[level]; m != nil {
		if v, ok := m[idx]; ok {
			return v
		}
	}
	bl := f.base.fieldLayers[level]
	if int(idx) < len(bl) && bl[idx] != nil {
		return *bl[idx]
	}
	return trie.ZeroHashes[level]
}

// recomputeOverlay walks up the trie from dirty leaves, hashing pairs
// and storing results in overrides. Returns the new root hash.
// dirtyLeaves maps leaf index → leaf hash at level 0.
func (f *FieldTrie) recomputeOverlay(dirtyLeaves map[uint64][32]byte) [32]byte {
	depth := len(f.overrides)
	hasher := hash.CustomSHA256Hasher()
	var combinedChunks [64]byte

	// Store dirty leaves in overrides[0].
	if f.overrides[0] == nil {
		f.overrides[0] = make(map[uint64][32]byte, len(dirtyLeaves))
	}
	for idx, h := range dirtyLeaves {
		f.overrides[0][idx] = h
	}

	// Walk up from level 0 to depth-1.
	currentDirty := dirtyLeaves
	for level := 0; level < depth-1; level++ {
		parentDirty := make(map[uint64][32]byte, len(currentDirty)/2+1)
		for idx := range currentDirty {
			parentIdx := idx / 2
			if _, done := parentDirty[parentIdx]; done {
				continue
			}
			leftIdx := parentIdx * 2
			rightIdx := leftIdx + 1

			left := f.readOverlayNode(level, leftIdx)
			right := f.readOverlayNode(level, rightIdx)

			copy(combinedChunks[:32], left[:])
			copy(combinedChunks[32:], right[:])
			parentHash := hasher(combinedChunks[:])

			parentDirty[parentIdx] = parentHash
			if f.overrides[level+1] == nil {
				f.overrides[level+1] = make(map[uint64][32]byte)
			}
			f.overrides[level+1][parentIdx] = parentHash
		}
		currentDirty = parentDirty
	}

	// The root is at overrides[depth-1][0], or fallback to base.
	return f.readOverlayNode(depth-1, 0)
}

// recomputeOverlayDispatch handles DataType-specific preprocessing before
// calling recomputeOverlay for the core walk-up.
func (f *FieldTrie) recomputeOverlayDispatch(indices []uint64, fieldRoots [][32]byte) ([32]byte, error) {
	// Check promotion threshold before adding to overrides.
	if len(indices) > OverlayPromotionThreshold || f.overlaySize() > OverlayPromotionThreshold {
		f.promoteToOwned()
		return f.recomputeOwned(indices, fieldRoots)
	}

	switch f.dataType {
	case types.BasicArray:
		dirtyLeaves := make(map[uint64][32]byte, len(indices))
		for i, idx := range indices {
			dirtyLeaves[idx] = fieldRoots[i]
		}
		root := f.recomputeOverlay(dirtyLeaves)
		return root, nil

	case types.CompositeArray:
		dirtyLeaves := make(map[uint64][32]byte, len(indices))
		for i, idx := range indices {
			dirtyLeaves[idx] = fieldRoots[i]
		}
		root := f.recomputeOverlay(dirtyLeaves)
		// Leaf count: max of base layer 0 len and any override indices.
		leafCount := uint64(len(f.base.fieldLayers[0]))
		for idx := range f.overrides[0] {
			if idx+1 > leafCount {
				leafCount = idx + 1
			}
		}
		return stateutil.AddInMixin(root, leafCount)

	case types.CompressedArray:
		numOfElems, err := f.field.ElemsInChunk()
		if err != nil {
			return [32]byte{}, err
		}
		// Deduplicate chunk indices.
		dirtyLeaves := make(map[uint64][32]byte, len(indices))
		for i, idx := range indices {
			chunkIdx := idx / numOfElems
			if _, exists := dirtyLeaves[chunkIdx]; !exists {
				dirtyLeaves[chunkIdx] = fieldRoots[i]
			}
		}
		root := f.recomputeOverlay(dirtyLeaves)
		return stateutil.AddInMixin(root, uint64(f.numOfElems))

	default:
		return [32]byte{}, errors.Errorf("unrecognized data type in field map: %v", reflect.TypeFor[types.DataType]().Name())
	}
}

// recomputeOwned handles trie recomputation for an owned-mode trie.
// This is the original RecomputeTrie logic, extracted for use after promotion.
func (f *FieldTrie) recomputeOwned(indices []uint64, fieldRoots [][32]byte) ([32]byte, error) {
	var fieldRoot [32]byte
	var err error
	switch f.dataType {
	case types.BasicArray:
		fieldRoot, f.fieldLayers, err = stateutil.RecomputeFromLayer(fieldRoots, indices, f.fieldLayers)
		if err != nil {
			return [32]byte{}, err
		}
		return fieldRoot, nil
	case types.CompositeArray:
		fieldRoot, f.fieldLayers, err = stateutil.RecomputeFromLayerVariable(fieldRoots, indices, f.fieldLayers)
		if err != nil {
			return [32]byte{}, err
		}
		return stateutil.AddInMixin(fieldRoot, uint64(len(f.fieldLayers[0])))
	case types.CompressedArray:
		numOfElems, err := f.field.ElemsInChunk()
		if err != nil {
			return [32]byte{}, err
		}
		iNumOfElems, err := pmath.Int(numOfElems)
		if err != nil {
			return [32]byte{}, err
		}
		var newIndices []uint64
		indexExists := make(map[uint64]bool)
		newRoots := make([][32]byte, 0, len(fieldRoots)/iNumOfElems)
		for i, idx := range indices {
			startIdx := idx / numOfElems
			if indexExists[startIdx] {
				continue
			}
			newIndices = append(newIndices, startIdx)
			indexExists[startIdx] = true
			newRoots = append(newRoots, fieldRoots[i])
		}
		fieldRoot, f.fieldLayers, err = stateutil.RecomputeFromLayerVariable(newRoots, newIndices, f.fieldLayers)
		if err != nil {
			return [32]byte{}, err
		}
		return stateutil.AddInMixin(fieldRoot, uint64(f.numOfElems))
	default:
		return [32]byte{}, errors.Errorf("unrecognized data type in field map: %v", reflect.TypeFor[types.DataType]().Name())
	}
}

// promoteToOwned converts an overlay trie to an owned trie by copying
// the base's layer data and applying all overrides on top.
func (f *FieldTrie) promoteToOwned() {
	baseLayers := f.base.fieldLayers
	owned := make([][]*[32]byte, len(baseLayers))
	for i, layer := range baseLayers {
		owned[i] = make([]*[32]byte, len(layer))
		copy(owned[i], layer)
	}
	// Apply all overrides on top of the copied layers.
	for level, m := range f.overrides {
		for idx, val := range m {
			v := val
			for int(idx) >= len(owned[level]) {
				zerohash := trie.ZeroHashes[level]
				owned[level] = append(owned[level], &zerohash)
			}
			owned[level][idx] = &v
		}
	}
	f.base.reference.MinusRef()
	f.fieldLayers = owned
	f.base = nil
	f.overrides = nil
}

// InsertFieldLayer manually inserts a field layer. This method
// bypasses the normal method of field computation, it is only
// meant to be used in tests.
func (f *FieldTrie) InsertFieldLayer(layer [][]*[32]byte) {
	f.fieldLayers = layer
}
