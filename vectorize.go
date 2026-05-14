package main

import (
	"time"
)

type Payload struct {
	ID          string      `json:"id"`
	Transaction Transaction `json:"transaction"`
	Customer    Customer    `json:"customer"`
	Merchant    Merchant    `json:"merchant"`
	Terminal    Terminal    `json:"terminal"`
	LastTx      *LastTx     `json:"last_transaction"`
}

type Transaction struct {
	Amount       float64 `json:"amount"`
	Installments float64 `json:"installments"`
	RequestedAt  string  `json:"requested_at"`
}

type Customer struct {
	AvgAmount      float64  `json:"avg_amount"`
	TxCount24h     float64  `json:"tx_count_24h"`
	KnownMerchants []string `json:"known_merchants"`
}

type Merchant struct {
	ID        string  `json:"id"`
	Mcc       string  `json:"mcc"`
	AvgAmount float64 `json:"avg_amount"`
}

type Terminal struct {
	IsOnline    bool    `json:"is_online"`
	CardPresent bool    `json:"card_present"`
	KmFromHome  float64 `json:"km_from_home"`
}

type LastTx struct {
	Timestamp     string  `json:"timestamp"`
	KmFromCurrent float64 `json:"km_from_current"`
}

// clamp restricts v to [0, 1].
func clamp(v float64) float64 {
	if v < 0 {
		return 0
	}
	if v > 1 {
		return 1
	}
	return v
}

// mondayBasedWeekday converts Go's Sun=0..Sat=6 to Mon=0..Sun=6.
func mondayBasedWeekday(t time.Time) int {
	return (int(t.Weekday()) + 6) % 7
}

// vectorize turns a payload into the 14-dimensional fraud-detection vector
//
// Returns ([14]float32{}, false) if the payload contains a malformed
// timestamp. Caller should respond with 400 Bad Request in that case.
func vectorize(p *Payload, cfg *Config, mcc map[string]float64) ([14]float32, bool) {
	reqAt, err := time.Parse(time.RFC3339, p.Transaction.RequestedAt)
	if err != nil {
		return [14]float32{}, false
	}
	reqAt = reqAt.UTC()

	var v [14]float32

	v[0] = float32(clamp(p.Transaction.Amount / cfg.MaxAmount))
	v[1] = float32(clamp(p.Transaction.Installments / cfg.MaxInstallments))

	// amount_vs_avg: guard against divide-by-zero on avg_amount=0
	if p.Customer.AvgAmount > 0 {
		v[2] = float32(clamp((p.Transaction.Amount / p.Customer.AvgAmount) / cfg.AmountVsAvgRatio))
	} else {
		v[2] = 1.0
	}

	v[3] = float32(float64(reqAt.Hour()) / 23.0)
	v[4] = float32(float64(mondayBasedWeekday(reqAt)) / 6.0)

	if p.LastTx == nil {
		v[5] = -1
		v[6] = -1
	} else {
		lastAt, err := time.Parse(time.RFC3339, p.LastTx.Timestamp)
		if err != nil {
			return [14]float32{}, false
		}
		minutes := reqAt.Sub(lastAt.UTC()).Minutes()
		v[5] = float32(clamp(minutes / cfg.MaxMinutes))
		v[6] = float32(clamp(p.LastTx.KmFromCurrent / cfg.MaxKm))
	}

	v[7] = float32(clamp(p.Terminal.KmFromHome / cfg.MaxKm))
	v[8] = float32(clamp(p.Customer.TxCount24h / cfg.MaxTxCount24h))

	if p.Terminal.IsOnline {
		v[9] = 1
	}
	if p.Terminal.CardPresent {
		v[10] = 1
	}

	// unknown_merchant: 1 if merchant.id is NOT in customer.known_merchants
	known := false
	for _, m := range p.Customer.KnownMerchants {
		if m == p.Merchant.ID {
			known = true
			break
		}
	}
	if !known {
		v[11] = 1
	}

	if r, ok := mcc[p.Merchant.Mcc]; ok {
		v[12] = float32(r)
	} else {
		v[12] = 0.5
	}

	v[13] = float32(clamp(p.Merchant.AvgAmount / cfg.MaxMerchantAvgAmount))

	return v, true
}
