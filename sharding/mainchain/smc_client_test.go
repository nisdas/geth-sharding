package mainchain

import (
	"testing"

	"github.com/ethereum/go-ethereum/sharding"
)

// Verifies that SMCClient implements the Client interface.
var _ = Client(&SMCClient{})

// Verifies that SMCCLient implements the sharding Service inteface.
var _ = sharding.Service(&SMCClient{})

func TestWaitForTransaction(t *testing.T) {
	client := &SMCClient{}

}
