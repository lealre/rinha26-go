package main

import (
	"encoding/json"
	"fmt"
	"os"
)

// Config holds the normalization constants from resources/normalization.json.
type Config struct {
	MaxAmount             float64 `json:"max_amount"`
	MaxInstallments       float64 `json:"max_installments"`
	AmountVsAvgRatio      float64 `json:"amount_vs_avg_ratio"`
	MaxMinutes            float64 `json:"max_minutes"`
	MaxKm                 float64 `json:"max_km"`
	MaxTxCount24h         float64 `json:"max_tx_count_24h"`
	MaxMerchantAvgAmount  float64 `json:"max_merchant_avg_amount"`
}

func loadConfig(path string) (*Config, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open normalization.json: %w", err)
	}
	defer f.Close()

	var c Config
	if err := json.NewDecoder(f).Decode(&c); err != nil {
		return nil, fmt.Errorf("decode normalization.json: %w", err)
	}
	return &c, nil
}

func loadMCCRisk(path string) (map[string]float64, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open mcc_risk.json: %w", err)
	}
	defer f.Close()

	m := map[string]float64{}
	if err := json.NewDecoder(f).Decode(&m); err != nil {
		return nil, fmt.Errorf("decode mcc_risk.json: %w", err)
	}
	return m, nil
}
