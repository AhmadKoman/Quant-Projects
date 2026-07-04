from AlgorithmImports import *
from datetime import datetime, timedelta
import csv
import math
import numpy as np


class CornState(PythonData):
    OBJECT_STORE_KEY = "corn_phase4_state_for_quantconnect.csv"
    HEADER = None

    @staticmethod
    def _clean(value):
        if value is None:
            return ""
        return str(value).strip()

    @staticmethod
    def _float(value, default=float("nan")):
        value = CornState._clean(value)
        if value == "" or value.lower() in ["nan", "none", "null"]:
            return default
        try:
            return float(value)
        except Exception:
            return default

    @staticmethod
    def _int(value, default=0):
        x = CornState._float(value, float("nan"))
        if not math.isfinite(x):
            return default
        return int(x)

    @staticmethod
    def _bool_int(value, default=False):
        value = CornState._clean(value)

        if value == "":
            return default

        lowered = value.lower()

        if lowered in ["true", "yes", "y"]:
            return True

        if lowered in ["false", "no", "n"]:
            return False

        return CornState._int(value, 1 if default else 0) == 1

    def get_source(self, config, date, is_live):
        return SubscriptionDataSource(
            CornState.OBJECT_STORE_KEY,
            SubscriptionTransportMedium.OBJECT_STORE,
            FileFormat.CSV,
        )

    def reader(self, config, line, date, is_live):
        if not line or not line.strip():
            return None

        try:
            cols = next(csv.reader([line.strip()]))
        except Exception:
            cols = line.strip().split(",")

        if not cols:
            return None

        if cols[0].strip().lower() == "date":
            CornState.HEADER = {
                name.strip(): i for i, name in enumerate(cols)
            }
            return None

        if CornState.HEADER is None:
            return None

        def get(name, default=""):
            idx = CornState.HEADER.get(name)
            if idx is None or idx >= len(cols):
                return default
            return cols[idx]

        def get_first(names, default=""):
            for name in names:
                value = get(name, "")
                if CornState._clean(value) != "":
                    return value
            return default

        try:
            row_date = datetime.strptime(get("date"), "%Y-%m-%d")
        except Exception:
            return None

        state = CornState()
        state.symbol = config.symbol
        state.time = row_date
        state.end_time = row_date + timedelta(days=1)

        state.front_settle = self._float(get("front_settle"))
        state.front_contract = get("front_contract")
        state.marketing_year = self._int(get("marketing_year"))
        state.marketing_year_day = self._int(get("marketing_year_day"))
        state.acreage_crop_year = self._int(get("acreage_crop_year"))
        state.seasonal_regime = get("seasonal_regime")

        state.front_return_1d_calc = self._float(get("front_return_1d_calc"))
        state.front_return_5d_calc = self._float(get("front_return_5d_calc"))
        state.front_realized_vol_20d_calc = self._float(
            get("front_realized_vol_20d_calc")
        )
        state.front_realized_vol_60d_calc = self._float(
            get("front_realized_vol_60d_calc")
        )

        state.prior_stocks_to_use_clean = self._float(
            get("prior_stocks_to_use_clean")
        )
        state.model_stocks_to_use_clean = self._float(
            get("model_stocks_to_use_clean")
        )
        state.stocks_to_use_change_clean = self._float(
            get("stocks_to_use_change_clean")
        )
        state.production_delta_bil_bu_clean = self._float(
            get("production_delta_bil_bu_clean")
        )
        state.managed_money_z_clean = self._float(
            get("managed_money_z_clean"),
            0.0,
        )

        state.myday_sin = self._float(get("myday_sin"))
        state.myday_cos = self._float(get("myday_cos"))
        state.scarcity_score = self._float(get("scarcity_score"))
        state.looseness_score = self._float(get("looseness_score"))

        state.supply_factor_z_clean = self._float(
            get_first(["supply_factor_z_clean", "supply_factor_z"])
        )
        state.balance_factor_z_clean = self._float(
            get_first(["balance_factor_z_clean", "balance_factor_z"])
        )
        state.demand_factor_z_clean = self._float(
            get_first(["demand_factor_z_clean", "demand_factor_z"])
        )
        state.positioning_factor_z_clean = self._float(
            get_first(["positioning_factor_z_clean", "positioning_factor_z"])
        )
        state.seasonality_factor_z_clean = self._float(
            get_first(["seasonality_factor_z_clean", "seasonality_factor_z"])
        )
        state.risk_factor_z_clean = self._float(
            get_first(["risk_factor_z_clean", "risk_factor_z"])
        )
        state.fundamental_tightness_anchor_z_clean = self._float(
            get_first([
                "fundamental_tightness_anchor_z_clean",
                "fundamental_tightness_anchor_z",
                "fundamental_tightness_factor_z_clean",
                "fundamental_tightness_factor_z",
            ])
        )

        state.phase3_factor_model_ready = self._bool_int(
            get_first(["phase3_factor_model_ready"]),
            default=False,
        )

        state.phase4_state_quality_score = self._int(
            get("phase4_state_quality_score")
        )
        state.phase4_state_quality_ok = self._bool_int(
            get("phase4_state_quality_ok")
        )
        state.phase4_outright_ready = self._bool_int(
            get("phase4_outright_ready"),
            default=True,
        )
        state.phase4_calendar_spread_ready = self._bool_int(
            get("phase4_calendar_spread_ready"),
            default=True,
        )
        state.phase4_basis_ready = self._bool_int(
            get("phase4_basis_ready"),
            default=False,
        )
        state.phase4_basis_reason = get("phase4_basis_reason")
        state.phase4_model_version = get("phase4_model_version")

        usable = (
            state.phase4_state_quality_ok
            and state.phase3_factor_model_ready
        )
        state.value = 1.0 if usable else 0.0

        return state


class RollingRidgeFactorCurveModel:
    def __init__(
        self,
        feature_names,
        min_rows=500,
        max_rows=5000,
        ridge_alpha=4.0,
        min_sigma=0.025,
    ):
        self.feature_names = list(feature_names)
        self.min_rows = min_rows
        self.max_rows = max_rows
        self.ridge_alpha = ridge_alpha
        self.min_sigma = min_sigma

        self.x_rows = []
        self.y_rows = []

        self.version = 0
        self.fit_version = None

        self.mu = None
        self.sd = None
        self.y_mu = None
        self.beta = None
        self.sigma = None
        self.fit_rows = 0

    def add(self, x, y):
        if x is None or not math.isfinite(y):
            return

        if len(x) != len(self.feature_names):
            return

        if not all(math.isfinite(v) for v in x):
            return

        self.x_rows.append([float(v) for v in x])
        self.y_rows.append(float(y))
        self.version += 1

        if len(self.y_rows) > self.max_rows:
            excess = len(self.y_rows) - self.max_rows
            self.x_rows = self.x_rows[excess:]
            self.y_rows = self.y_rows[excess:]
            self.version += 1

    @property
    def rows(self):
        return len(self.y_rows)

    def fit(self):
        if self.fit_version == self.version:
            return self.beta is not None

        if len(self.y_rows) < self.min_rows:
            return False

        X = np.array(self.x_rows, dtype=float)
        y = np.array(self.y_rows, dtype=float)

        good = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
        X = X[good]
        y = y[good]

        if len(y) < self.min_rows:
            return False

        mu = np.mean(X, axis=0)
        sd = np.std(X, axis=0)
        sd[sd < 1e-8] = 1.0

        Xs = (X - mu) / sd

        y_mu = float(np.mean(y))
        yc = y - y_mu

        A = Xs.T @ Xs + self.ridge_alpha * np.eye(Xs.shape[1])
        b = Xs.T @ yc

        try:
            beta = np.linalg.solve(A, b)
        except Exception:
            beta = np.linalg.pinv(A) @ b

        fitted = y_mu + Xs @ beta
        sigma = float(np.std(y - fitted))
        sigma = max(sigma, self.min_sigma)

        self.mu = mu
        self.sd = sd
        self.y_mu = y_mu
        self.beta = beta
        self.sigma = sigma
        self.fit_rows = len(y)
        self.fit_version = self.version

        return True

    def predict(self, x):
        if x is None:
            return None

        if len(x) != len(self.feature_names):
            return None

        if not all(math.isfinite(v) for v in x):
            return None

        if not self.fit():
            return None

        x0 = (np.array(x, dtype=float).reshape(1, -1) - self.mu) / self.sd
        yhat = float(self.y_mu + x0 @ self.beta)

        return yhat, self.sigma, self.fit_rows

    def beta_by_name(self):
        if self.beta is None:
            return {}

        return {
            name: float(beta)
            for name, beta in zip(self.feature_names, self.beta)
        }


class CornPhase4FactorCurveAndSpreadAlgorithm(QCAlgorithm):
    def initialize(self):
        self.set_start_date(2022, 1, 5)
        self.set_end_date(2024, 4, 23)
        self.set_cash(250000)

        self.min_days_to_expiry = 7
        self.max_days_to_expiry = 540
        self.max_depth = 7

        self.entry_z_outright = 0.90
        self.entry_z_spread = 0.70
        self.exit_z = 0.30

        self.contracts_per_outright = 1
        self.contracts_per_spread_leg = 1
        self.max_abs_contracts_per_symbol = 2
        self.max_state_age_days = 7

        self.allow_outrights = True
        self.allow_spreads = True
        self.require_phase3_factors = True

        self.spread_preference_multiplier = 0.90

        self.phase4_symbol = self.add_data(
            CornState,
            "CORN_PHASE4_STATE",
            Resolution.DAILY,
        ).symbol

        self.corn_future = self.add_future(
            Futures.Grains.CORN,
            Resolution.DAILY,
            data_mapping_mode=DataMappingMode.OPEN_INTEREST,
            data_normalization_mode=DataNormalizationMode.BACKWARDS_RATIO,
            contract_depth_offset=0,
        )
        self.corn_future.set_filter(0, self.max_days_to_expiry)
        self.corn_symbol = self.corn_future.symbol

        self.feature_names = self._feature_names()

        self.factor_curve_model = RollingRidgeFactorCurveModel(
            feature_names=self.feature_names,
            min_rows=500,
            max_rows=5000,
            ridge_alpha=4.0,
            min_sigma=0.025,
        )

        self.latest_state = None
        self.last_train_date = None
        self.current_signal_key = None
        self.current_mode = "flat"

        self.last_reject_debug_date = None

        self.history_warmup_days = 730
        self.set_warm_up(
            timedelta(days=self.history_warmup_days),
            Resolution.DAILY,
        )

        self.debug(
            "Initialized Corn Phase 4 factor curve strategy with warmup"
        )

    def _feature_names(self):
        return [
            "supply_factor",
            "balance_factor",
            "demand_factor",
            "positioning_factor",
            "seasonality_factor",
            "risk_factor",
            "fundamental_anchor",
            "stocks_to_use",
            "stocks_to_use_change",
            "production_delta",
            "myday_sin",
            "myday_cos",
            "dte_years",
            "sqrt_dte",
            "month_sin",
            "month_cos",
            "depth_scaled",
            "supply_x_dte",
            "balance_x_dte",
            "demand_x_dte",
            "seasonality_x_dte",
            "anchor_x_dte",
            "balance_x_month_sin",
            "balance_x_month_cos",
            "seasonality_x_month_sin",
            "seasonality_x_month_cos",
        ]

    def on_warmup_finished(self):
        self.debug(
            f"Warmup finished | rows={self.factor_curve_model.rows} "
            f"| min_rows={self.factor_curve_model.min_rows}"
        )

    def on_data(self, data):
        if data.contains_key(self.phase4_symbol):
            self.latest_state = data[self.phase4_symbol]

        state = self.latest_state
        if state is None:
            return

        chain = data.futures_chains.get(self.corn_symbol)
        if not chain:
            return

        contracts = self._get_clean_chain_contracts(chain)
        if len(contracts) < 2:
            return

        state_age = (self.time.date() - state.time.date()).days

        if state_age > self.max_state_age_days:
            if not self.is_warming_up:
                self._set_targets({}, "stale_phase4_state")
            return

        if not self._state_is_usable(state):
            self._debug_state_reject_reason(state)
            if not self.is_warming_up:
                self._set_targets({}, "bad_or_unready_phase4_state")
            return

        self._plot_factor_inputs(state)

        if self.is_warming_up:
            self._train_after_decision(state, contracts)
            return

        valued_contracts = self._estimate_contract_fair_values(
            state,
            contracts,
        )

        spread_opportunities = self._estimate_calendar_spreads(
            valued_contracts,
        )

        self._phase5_trade(
            state,
            valued_contracts,
            spread_opportunities,
        )

        self._train_after_decision(state, contracts)

    def _state_is_usable(self, state):
        if state is None:
            return False

        if not state.phase4_state_quality_ok:
            return False

        if self.require_phase3_factors:
            if not state.phase3_factor_model_ready:
                return False

        required_values = [
            state.supply_factor_z_clean,
            state.balance_factor_z_clean,
            state.demand_factor_z_clean,
            state.positioning_factor_z_clean,
            state.seasonality_factor_z_clean,
            state.risk_factor_z_clean,
            state.fundamental_tightness_anchor_z_clean,
            state.model_stocks_to_use_clean,
            state.stocks_to_use_change_clean,
            state.production_delta_bil_bu_clean,
            state.myday_sin,
            state.myday_cos,
        ]

        for value in required_values:
            if not math.isfinite(value):
                return False

        return True

    def _debug_state_reject_reason(self, state):
        current_date = self.time.date()

        if self.last_reject_debug_date == current_date:
            return

        self.last_reject_debug_date = current_date

        if state is None:
            self.debug(f"{current_date} rejected state: state is None")
            return

        checks = [
            ("phase4_state_quality_ok", state.phase4_state_quality_ok),
            ("phase3_factor_model_ready", state.phase3_factor_model_ready),
            ("supply_factor_z_clean", state.supply_factor_z_clean),
            ("balance_factor_z_clean", state.balance_factor_z_clean),
            ("demand_factor_z_clean", state.demand_factor_z_clean),
            ("positioning_factor_z_clean", state.positioning_factor_z_clean),
            ("seasonality_factor_z_clean", state.seasonality_factor_z_clean),
            ("risk_factor_z_clean", state.risk_factor_z_clean),
            (
                "fundamental_tightness_anchor_z_clean",
                state.fundamental_tightness_anchor_z_clean,
            ),
            ("model_stocks_to_use_clean", state.model_stocks_to_use_clean),
            (
                "stocks_to_use_change_clean",
                state.stocks_to_use_change_clean,
            ),
            (
                "production_delta_bil_bu_clean",
                state.production_delta_bil_bu_clean,
            ),
            ("myday_sin", state.myday_sin),
            ("myday_cos", state.myday_cos),
        ]

        bad = []

        for name, value in checks:
            if isinstance(value, bool):
                if not value:
                    bad.append(name)
            else:
                try:
                    x = float(value)
                except Exception:
                    bad.append(name)
                    continue

                if not math.isfinite(x):
                    bad.append(name)

        if bad:
            self.debug(
                f"{current_date} rejected state, missing or bad: "
                + ", ".join(bad[:8])
            )

    def _get_clean_chain_contracts(self, chain):
        rows = []

        for contract in chain:
            symbol = contract.symbol
            expiry = contract.expiry
            days_to_expiry = (expiry.date() - self.time.date()).days

            if days_to_expiry < self.min_days_to_expiry:
                continue

            if days_to_expiry > self.max_days_to_expiry:
                continue

            price = self._get_contract_price(contract)
            if not math.isfinite(price) or price <= 0:
                continue

            rows.append({
                "symbol": symbol,
                "expiry": expiry,
                "days_to_expiry": days_to_expiry,
                "month": expiry.month,
                "price": price,
                "open_interest": self._safe_float(
                    getattr(contract, "open_interest", float("nan"))
                ),
            })

        rows = sorted(rows, key=lambda r: (r["expiry"], str(r["symbol"])))
        rows = rows[: self.max_depth]

        for i, row in enumerate(rows):
            row["depth"] = i + 1
            row["role"] = f"C{i + 1}"

        return rows

    def _get_contract_price(self, contract):
        price = self._safe_float(
            getattr(contract, "last_price", float("nan"))
        )

        if math.isfinite(price) and price > 0:
            return price

        symbol = contract.symbol

        if self.securities.contains_key(symbol):
            sec_price = self._safe_float(self.securities[symbol].price)
            if math.isfinite(sec_price) and sec_price > 0:
                return sec_price

        return float("nan")

    @staticmethod
    def _safe_float(x, default=float("nan")):
        try:
            return float(x)
        except Exception:
            return default

    def _feature_vector(self, state, contract_row):
        dte = float(contract_row["days_to_expiry"])
        dte_years = dte / 365.25
        sqrt_dte = math.sqrt(max(dte, 0.0)) / math.sqrt(365.25)

        month = float(contract_row["month"])
        month_sin = math.sin(2.0 * math.pi * month / 12.0)
        month_cos = math.cos(2.0 * math.pi * month / 12.0)

        depth_scaled = float(contract_row["depth"]) / float(self.max_depth)

        supply = state.supply_factor_z_clean
        balance = state.balance_factor_z_clean
        demand = state.demand_factor_z_clean
        positioning = state.positioning_factor_z_clean
        seasonality = state.seasonality_factor_z_clean
        risk = state.risk_factor_z_clean
        anchor = state.fundamental_tightness_anchor_z_clean

        stu = state.model_stocks_to_use_clean
        stu_change = state.stocks_to_use_change_clean
        prod_delta = state.production_delta_bil_bu_clean

        myday_sin = state.myday_sin
        myday_cos = state.myday_cos

        x = [
            supply,
            balance,
            demand,
            positioning,
            seasonality,
            risk,
            anchor,
            stu,
            stu_change,
            prod_delta,
            myday_sin,
            myday_cos,
            dte_years,
            sqrt_dte,
            month_sin,
            month_cos,
            depth_scaled,
            supply * dte_years,
            balance * dte_years,
            demand * dte_years,
            seasonality * dte_years,
            anchor * dte_years,
            balance * month_sin,
            balance * month_cos,
            seasonality * month_sin,
            seasonality * month_cos,
        ]

        if len(x) != len(self.feature_names):
            return None

        if not all(math.isfinite(v) for v in x):
            return None

        return x

    def _estimate_contract_fair_values(self, state, contracts):
        valued = []

        for row in contracts:
            x = self._feature_vector(state, row)
            prediction = self.factor_curve_model.predict(x)

            if prediction is None:
                continue

            fair_log, sigma_log, train_rows = prediction

            fair_value = math.exp(fair_log)
            market_log = math.log(row["price"])
            z = (market_log - fair_log) / sigma_log

            item = dict(row)
            item.update({
                "features": x,
                "fair_log": fair_log,
                "fair_value": fair_value,
                "sigma_log": sigma_log,
                "train_rows": train_rows,
                "outright_z": z,
                "outright_error": row["price"] - fair_value,
            })

            valued.append(item)

        return valued

    def _estimate_calendar_spreads(self, valued_contracts):
        by_depth = {row["depth"]: row for row in valued_contracts}

        pairs = [
            (1, 2),
            (1, 3),
            (1, 4),
            (2, 3),
            (2, 4),
            (2, 5),
            (3, 4),
            (3, 5),
            (4, 6),
            (5, 7),
        ]

        spreads = []

        for near_depth, far_depth in pairs:
            if near_depth not in by_depth or far_depth not in by_depth:
                continue

            near = by_depth[near_depth]
            far = by_depth[far_depth]

            market_spread = near["price"] - far["price"]
            fair_spread = near["fair_value"] - far["fair_value"]

            near_unc_price = near["fair_value"] * near["sigma_log"] * 0.50
            far_unc_price = far["fair_value"] * far["sigma_log"] * 0.50

            spread_unc = math.sqrt(
                near_unc_price ** 2 + far_unc_price ** 2
            )
            spread_unc = max(spread_unc, 4.0)

            spread_z = (market_spread - fair_spread) / spread_unc
            days_between = (
                far["expiry"].date() - near["expiry"].date()
            ).days

            if days_between <= 0:
                continue

            spreads.append({
                "key": f"SPREAD_C{near_depth}_C{far_depth}",
                "near": near,
                "far": far,
                "near_depth": near_depth,
                "far_depth": far_depth,
                "market_spread": market_spread,
                "fair_spread": fair_spread,
                "spread_mispricing": market_spread - fair_spread,
                "spread_uncertainty": spread_unc,
                "spread_z": spread_z,
                "days_between": days_between,
            })

        return spreads

    def _train_after_decision(self, state, contracts):
        current_date = self.time.date()

        if self.last_train_date == current_date:
            return

        if not self._state_is_usable(state):
            return

        for row in contracts:
            x = self._feature_vector(state, row)
            price = row["price"]

            if x is None:
                continue

            if not math.isfinite(price) or price <= 0:
                continue

            self.factor_curve_model.add(x, math.log(price))

        self.last_train_date = current_date

    def _phase5_trade(self, state, valued_contracts, spread_opportunities):
        if self.factor_curve_model.rows < self.factor_curve_model.min_rows:
            self._set_targets(
                {},
                f"warming_up_factor_model_rows={self.factor_curve_model.rows}",
            )
            return

        opportunities = []

        if self.allow_outrights and state.phase4_outright_ready:
            for row in valued_contracts:
                if row["depth"] > 3:
                    continue

                z = row["outright_z"]

                opportunities.append({
                    "type": "outright",
                    "key": f"OUTRIGHT_C{row['depth']}",
                    "score": abs(z),
                    "z": z,
                    "row": row,
                })

        if self.allow_spreads and state.phase4_calendar_spread_ready:
            for sp in spread_opportunities:
                z = sp["spread_z"]

                opportunities.append({
                    "type": "spread",
                    "key": sp["key"],
                    "score": abs(z),
                    "z": z,
                    "spread": sp,
                })

        if not opportunities:
            self._set_targets({}, "no_phase4_opportunities")
            return

        best_outright = self._best([
            o for o in opportunities if o["type"] == "outright"
        ])
        best_spread = self._best([
            o for o in opportunities if o["type"] == "spread"
        ])

        chosen = None

        if best_spread and best_spread["score"] >= self.entry_z_spread:
            chosen = best_spread

        if best_outright and best_outright["score"] >= self.entry_z_outright:
            if chosen is None:
                chosen = best_outright
            elif (
                best_outright["score"]
                > chosen["score"] / self.spread_preference_multiplier
            ):
                chosen = best_outright

        if chosen is None and self.current_signal_key is not None:
            current = next(
                (
                    o for o in opportunities
                    if o["key"] == self.current_signal_key
                ),
                None,
            )

            if current and current["score"] > self.exit_z:
                chosen = current

        if chosen is None:
            self.current_signal_key = None
            self.current_mode = "flat"
            self._set_targets({}, "no_signal_or_exit")
            self._plot_diagnostics(best_outright, best_spread, None)
            return

        targets, tag = self._targets_from_opportunity(chosen)

        self.current_signal_key = chosen["key"]
        self.current_mode = chosen["type"]

        self._set_targets(targets, tag)
        self._plot_diagnostics(best_outright, best_spread, chosen)

    @staticmethod
    def _best(items):
        if not items:
            return None

        return sorted(
            items,
            key=lambda o: o["score"],
            reverse=True,
        )[0]

    def _targets_from_opportunity(self, op):
        z = op["z"]

        if op["type"] == "outright":
            row = op["row"]

            qty = (
                -self.contracts_per_outright
                if z > 0
                else self.contracts_per_outright
            )
            qty = self._clip_qty(qty)

            targets = {row["symbol"]: qty}

            tag = (
                f"Phase5 OUTRIGHT {row['role']} z={z:.2f} "
                f"market={row['price']:.2f} fair={row['fair_value']:.2f} "
                f"train={row['train_rows']}"
            )

            return targets, tag

        sp = op["spread"]
        near = sp["near"]
        far = sp["far"]

        near_qty = (
            -self.contracts_per_spread_leg
            if z > 0
            else self.contracts_per_spread_leg
        )

        far_qty = (
            self.contracts_per_spread_leg
            if z > 0
            else -self.contracts_per_spread_leg
        )

        targets = {
            near["symbol"]: self._clip_qty(near_qty),
            far["symbol"]: self._clip_qty(far_qty),
        }

        tag = (
            f"Phase5 SPREAD C{sp['near_depth']}-C{sp['far_depth']} "
            f"z={z:.2f} market={sp['market_spread']:.2f} "
            f"fair={sp['fair_spread']:.2f} "
            f"mis={sp['spread_mispricing']:.2f}"
        )

        return targets, tag

    def _clip_qty(self, qty):
        return int(
            max(
                -self.max_abs_contracts_per_symbol,
                min(self.max_abs_contracts_per_symbol, qty),
            )
        )

    def _set_targets(self, targets, tag):
        target_symbols = set(targets.keys())

        for symbol, security in self.securities.items():
            if symbol.security_type != SecurityType.FUTURE:
                continue

            if symbol not in target_symbols and self.portfolio[symbol].invested:
                self.liquidate(symbol, tag=f"Flatten | {tag}")

        for symbol, target_qty in targets.items():
            current_qty = int(self.portfolio[symbol].quantity)
            order_qty = int(target_qty - current_qty)

            if order_qty != 0:
                self.market_order(symbol, order_qty, tag=tag)

    def _plot_factor_inputs(self, state):
        if self.time.day % 5 != 0:
            return

        self.plot("Factor Inputs", "supply", state.supply_factor_z_clean)
        self.plot("Factor Inputs", "balance", state.balance_factor_z_clean)
        self.plot("Factor Inputs", "demand", state.demand_factor_z_clean)
        self.plot(
            "Factor Inputs",
            "positioning",
            state.positioning_factor_z_clean,
        )
        self.plot(
            "Factor Inputs",
            "seasonality",
            state.seasonality_factor_z_clean,
        )
        self.plot("Factor Inputs", "risk", state.risk_factor_z_clean)
        self.plot(
            "Factor Inputs",
            "anchor",
            state.fundamental_tightness_anchor_z_clean,
        )

    def _plot_diagnostics(self, best_outright, best_spread, chosen):
        if best_outright:
            self.plot("Phase4", "best_outright_z", best_outright["z"])
            self.plot(
                "Signals",
                "best_outright_abs_z",
                best_outright["score"],
            )

        if best_spread:
            self.plot("Phase4", "best_spread_z", best_spread["z"])
            self.plot(
                "Signals",
                "best_spread_abs_z",
                best_spread["score"],
            )

        if chosen:
            mode_value = 1 if chosen["type"] == "outright" else 2
            self.plot("Phase5", "chosen_abs_z", chosen["score"])
            self.plot("Phase5", "mode", mode_value)
        else:
            self.plot("Phase5", "chosen_abs_z", 0)
            self.plot("Phase5", "mode", 0)

        self.plot(
            "Phase4",
            "training_rows",
            self.factor_curve_model.rows,
        )

        if self.factor_curve_model.sigma is not None:
            self.plot(
                "Phase4",
                "residual_sigma",
                self.factor_curve_model.sigma,
            )

        self._plot_factor_betas()

    def _plot_factor_betas(self):
        if self.time.day % 5 != 0:
            return

        beta = self.factor_curve_model.beta_by_name()

        names = [
            "supply_factor",
            "balance_factor",
            "demand_factor",
            "positioning_factor",
            "seasonality_factor",
            "risk_factor",
            "fundamental_anchor",
        ]

        for name in names:
            if name in beta:
                self.plot("Factor Betas", name, beta[name])