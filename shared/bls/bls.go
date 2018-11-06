// Package bls implements a go-wrapper around a C BLS library leveraging
// the BLS12-381 curve. This package exposes a public API for verifying and
// aggregating BLS signatures used by Ethereum 2.0.
package bls

import (
	"fmt"
	"math/big"
)

// Signature used in the BLS signature scheme.
type Signature struct{}

// SecretKey used in the BLS scheme.
type SecretKey struct {
	K *big.Int
}

// PublicKey corresponding to secret key used in the BLS scheme.
type PublicKey struct{}

// PublicKey returns the corresponding public key for the
// Secret Key
func (s *SecretKey) PublicKey() (*PublicKey, error) {
	return &PublicKey{}, nil
}

func (s *SecretKey) BufferedSecretKey() []byte {
	return s.K.Bytes()
}

func (p *PublicKey) BufferedPublicKey() []byte {
	return []byte{}
}

func (s *SecretKey) UnBufferSecretKey(bufferedKey []byte) {
	s.K = big.NewInt(0).SetBytes(bufferedKey)

}

func (p *PublicKey) UnBufferPublicKey(bufferedKey []byte) {

}

func GenerateKey(seed []byte) *SecretKey {
	return &SecretKey{
		K: big.NewInt(0).SetBytes(seed),
	}
}

// Sign a message using a secret key - in a beacon/validator client,
// this key will come from and be unlocked from the account keystore.
func Sign(sec *SecretKey, msg []byte) (*Signature, error) {
	return &Signature{}, nil
}

// VerifySig against a public key.
func VerifySig(pub *PublicKey, msg []byte, sig *Signature) (bool, error) {
	return true, nil
}

// VerifyAggregateSig created using the underlying BLS signature
// aggregation scheme.
func VerifyAggregateSig(pubs []*PublicKey, msg []byte, asig *Signature) (bool, error) {
	return true, nil
}

// BatchVerify a list of individual signatures by aggregating them.
func BatchVerify(pubs []*PublicKey, msg []byte, sigs []*Signature) (bool, error) {
	asigs, err := AggregateSigs(sigs)
	if err != nil {
		return false, fmt.Errorf("could not aggregate signatures: %v", err)
	}
	return VerifyAggregateSig(pubs, msg, asigs)
}

// AggregateSigs puts multiple signatures into one using the underlying
// BLS sum functions.
func AggregateSigs(sigs []*Signature) (*Signature, error) {
	return &Signature{}, nil
}
