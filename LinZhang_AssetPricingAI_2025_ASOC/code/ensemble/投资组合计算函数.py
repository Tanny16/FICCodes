import os
import os.path as op
import pdb

import numpy as np
import pandas as pd


class PortfolioManager(object):
    def __init__(
            self,
            signal_df: pd.DataFrame,
            freq: str,
            portfolio_dir: str,
            start_year=1987,
            end_year=2016,
            delay_list=None,  # 默认为 None，后续会自动设为 [0]
            load_signal=True,
            custom_ret=None,
            transaction_cost=False,
    ):
        assert freq in ["week", "month", "quarter"]
        self.freq = freq
        self.portfolio_dir = portfolio_dir
        self.start_year = start_year
        self.end_year = end_year
        self.no_delay_ret_name = f"next_{freq}_ret"
        self.custom_ret = custom_ret
        self.delay_list = [0] if delay_list is None else delay_list  # 默认为 [0]
        self.transaction_cost = transaction_cost
        if load_signal:
            assert "up_prob" in signal_df.columns
            self.signal_df = self.get_up_prob_with_period_ret(signal_df)

    def get_up_prob_with_period_ret(self, signal_df):
        # 直接使用输入的 signal_df
        signal_df = signal_df[
            signal_df.index.get_level_values("ending_date").year.isin(
                range(self.start_year, self.end_year + 1)
            )
        ]

        # 保留 signal_df 中已有的 MarketCap 和 ret_val 列
        signal_df.MarketCap = signal_df.MarketCap.abs()
        signal_df = signal_df[~signal_df.MarketCap.isnull()].copy()

        # 无需再添加 ret_val 这类列
        signal_df[self.no_delay_ret_name] = signal_df["ret_val"]
        return signal_df


    def calculate_portfolio_rets(self, weight_type, cut=10, delay=0):
        assert weight_type in ["ew", "vw"]
        assert delay == 0  # 只支持 delay = 0
        if self.custom_ret:
            print(f"Calculating portfolio using {self.custom_ret}")
            ret_name = self.custom_ret
        else:
            ret_name = self.no_delay_ret_name

        df = self.signal_df.copy()

        def __get_decile_df_with_inv_ret(reb_df, decile_idx):
            rebalance_up_prob = reb_df.up_prob
            low = np.percentile(rebalance_up_prob, decile_idx * 100.0 / cut)
            high = np.percentile(rebalance_up_prob, (decile_idx + 1) * 100.0 / cut)
            if decile_idx == 0:
                pf_filter = (rebalance_up_prob >= low) & (rebalance_up_prob <= high)
            else:
                pf_filter = (rebalance_up_prob > low) & (rebalance_up_prob <= high)
            _decile_df = reb_df[pf_filter].copy()
            if weight_type == "ew":
                stock_num = len(_decile_df)
                _decile_df["weight"] = (
                        pd.Series(
                            np.ones(stock_num), dtype=np.float64, index=_decile_df.index
                        )
                        / stock_num
                )
            else:
                value = _decile_df.MarketCap
                _decile_df["weight"] = pd.Series(value, dtype=np.float64) / np.sum(value)
            _decile_df["inv_ret"] = _decile_df["weight"] * _decile_df[ret_name]
            return _decile_df

        dates = np.sort(np.unique(df.index.get_level_values("ending_date")))
        print(f"Calculating portfolio from {dates[0]}, {dates[1]} to {dates[-1]}")
        turnover = np.zeros(len(dates) - 1)
        portfolio_ret = pd.DataFrame(index=dates, columns=list(range(cut)))
        prev_to_df = None
        #prob_ret_corr = []
        #prob_ret_pearson_corr = []
        prob_inv_ret_corr = []
        prob_inv_ret_pearson_corr = []

        for i, date in enumerate(dates):
            rebalance_df = df.loc[date].copy()
            """rank_corr = ut.rank_corr(
                rebalance_df, "up_prob", ret_name, method="spearman"
            )
            pearson_corr = ut.rank_corr(
                rebalance_df, "up_prob", ret_name, method="pearson"
            )
            prob_ret_corr.append(rank_corr)
            prob_ret_pearson_corr.append(pearson_corr)"""

            low = np.percentile(rebalance_df.up_prob, 10)
            high = np.percentile(rebalance_df.up_prob, 90)
            if low == high:
                print(f"Skipping {date}")
                continue
            for j in range(cut):
                decile_df = __get_decile_df_with_inv_ret(rebalance_df, j)
                if self.transaction_cost:
                    if j == cut - 1:
                        decile_df["inv_ret"] -= (
                                decile_df["weight"] * decile_df["transaction_fee"] * 2
                        )
                    elif j == 0:
                        decile_df["inv_ret"] += (
                                decile_df["weight"] * decile_df["transaction_fee"] * 2
                        )
                portfolio_ret.loc[date, j] = np.sum(decile_df["inv_ret"])

            sell_decile = __get_decile_df_with_inv_ret(rebalance_df, 0)
            buy_decile = __get_decile_df_with_inv_ret(rebalance_df, cut - 1)
            buy_sell_decile = pd.concat([sell_decile, buy_decile])

            """prob_inv_ret_corr.append(
                ut.rank_corr(buy_sell_decile, "up_prob", "inv_ret", method="spearman")
            )
            prob_inv_ret_pearson_corr.append(
                ut.rank_corr(buy_sell_decile, "up_prob", "inv_ret", method="pearson")
            )"""

            sell_decile[["weight", "inv_ret"]] = sell_decile[["weight", "inv_ret"]] * (
                -1
            )
            to_df = pd.concat([sell_decile, buy_decile])

            if i > 0:
                tto_df = pd.DataFrame(
                    index=np.unique(list(to_df.index) + list(prev_to_df.index))
                )
                try:
                    tto_df["cur_weight"] = to_df["weight"]
                except ValueError:
                    pdb.set_trace()
                tto_df[["prev_weight", "ret", "inv_ret"]] = prev_to_df[["weight", ret_name, "inv_ret"]]
                tto_df.fillna(0, inplace=True)
                denom = 1 + np.sum(tto_df["inv_ret"])
                turnover[i - 1] = np.sum(
                    (
                            tto_df["cur_weight"]
                            - tto_df["prev_weight"] * (1 + tto_df["ret"]) / denom
                    ).abs()
                )
                turnover[i - 1] = turnover[i - 1] * 0.5
            prev_to_df = to_df

        portfolio_ret = portfolio_ret.fillna(0)
        portfolio_ret["H-L"] = portfolio_ret[cut - 1] - portfolio_ret[0]
        """print(
            f"Spearman Corr between Prob and Stock Return is {np.nanmean(prob_ret_corr):.4f}"
        )
        print(
            f"Pearson Corr between Prob and Stock Return is {np.nanmean(prob_ret_pearson_corr):.4f}"
        )
        print(
            f"Spearman Corr between Prob and Top/Bottom deciles Return is {np.nanmean(prob_inv_ret_corr):.4f}"
        )
        print(
            f"Pearson Corr between Prob and Top/Bottom deciles Return is {np.nanmean(prob_inv_ret_pearson_corr):.4f}"
        )"""
        return portfolio_ret, np.mean(turnover)

    @staticmethod
    def _ret_to_cum_log_ret(rets):
        log_rets = np.log(rets.astype(float) + 1)
        return log_rets.cumsum()

    def make_portfolio_plot(
            self, portfolio_ret, cut, weight_type, save_path, plot_title
    ):
        ret_name = "nxt_freq_ewret" if weight_type == "ew" else "nxt_freq_vwret"
        df = portfolio_ret.copy()
        df.columns = ["Low(L)"] + [str(i) for i in range(2, cut)] + ["High(H)", "H-L"]
        spy = eqd.get_spy_freq_rets(self.freq)
        df["SPY"] = spy[ret_name]
        df.dropna(inplace=True)
        top_col_name, bottom_col_name = f"Top {cut}", f"Bottom {cut}"

        fig, ax = plt.subplots(figsize=(8, 6))
        df[[top_col_name, bottom_col_name, "SPY"]].cumsum().plot(ax=ax)
        ax.set_title(plot_title)
        plt.legend(loc="upper left")
        plt.savefig(save_path)
        plt.close()
        print(f"Portfolio plot saved to {save_path}")

    def portfolio_res_summary(self, portfolio_ret, turnover, cut=10):
        avg = portfolio_ret.mean().to_numpy()
        std = portfolio_ret.std().to_numpy()
        res = np.zeros((cut + 1, 3))
        period = 52 if self.freq == "week" else 12 if self.freq == "month" else 4
        res[:, 0] = avg * period
        res[:, 1] = std * np.sqrt(period)
        res[:, 2] = res[:, 0] / res[:, 1]

        summary_df = pd.DataFrame(res, columns=["ret", "std", "SR"])
        summary_df = summary_df.set_index(
            pd.Index(["Low"] + list(range(2, int(cut))) + ["High", "H-L"])
        )
        summary_df.loc["Turnover", "SR"] = turnover / (
            1 / 4
            if self.freq == "week"
            else 1
            if self.freq == "month"
            else 3
            if self.freq == "quarter"
            else 12
        )
        print(summary_df)
        return summary_df

    def generate_portfolio(
            self, weight_type, cut=10, delay=0, **kwargs
    ):
        assert weight_type in ["ew", "vw"]
        assert delay == 0  # 只支持 delay = 0
        pf_name = self.get_portfolio_name(weight_type, cut, delay)

        portfolio_ret, turnover = self.calculate_portfolio_rets(
            weight_type, cut, delay
        )

        data_dir = os.path.abspath(os.path.join(self.portfolio_dir, "pf_data"))
        print("portfolio_dir:", self.portfolio_dir)
        print("data_dir:", data_dir)

            # 确保文件夹存在
        os.makedirs(data_dir, exist_ok=True)

        csv_path = os.path.join(data_dir, f"pf_data_{pf_name}.csv")
        print("CSV file path:", csv_path)
        print("Portfolio Data:", portfolio_ret.head())  # 打印部分数据，确保数据格式正确

        try:
            portfolio_ret.to_csv(csv_path)
            print(f"CSV file saved successfully: {csv_path}")
        except Exception as e:
            print(f"Error saving CSV file: {e}")

        summary_df = self.portfolio_res_summary(portfolio_ret, turnover, cut)
        smry_path = os.path.join(self.portfolio_dir, f"{pf_name}.csv")
        print("smry_path:", smry_path)

            # 保存汇总文件
        try:
            summary_df.to_csv(smry_path)
            print(f"Summary saved to {smry_path}")
        except Exception as e:
            print(f"Error saving summary CSV file: {e}")




    def get_portfolio_name(self, weight_type, cut, delay):
        assert weight_type.lower() in ["ew", "vw"]
        delay_prefix = "" if delay == 0 else f"{delay}d_delay_"
        cut_surfix = "" if cut == 10 else f"_{cut}cut"
        custom_ret_surfix = "" if self.custom_ret is None else self.custom_ret
        tc_surfix = "" if not self.transaction_cost else "_w_transaction_cost"
        pf_name = f"{delay_prefix}{weight_type.lower()}{cut_surfix}{custom_ret_surfix}{tc_surfix}"
        return pf_name

    def load_portfolio_ret(self, weight_type, cut=10, delay=0):
        pf_name = self.get_portfolio_name(weight_type, cut, delay)
        try:
            df = pd.read_csv(
                op.join(self.portfolio_dir, "pf_data", f"pf_data_{pf_name}.csv"),
                index_col=0,
            )
        except FileNotFoundError:
            df = pd.read_csv(
                op.join(self.portfolio_dir, "pf_data", f"pf_data_{pf_name}_100.csv"),
                index_col=0,
            )
        df.index = pd.to_datetime(df.index)
        return df

    def load_portfolio_summary(self, weight_type, cut=10, delay=0):
        pf_name = self.get_portfolio_name(weight_type, cut, delay)
        try:
            df = pd.read_csv(op.join(self.portfolio_dir, f"{pf_name}.csv"), index_col=0)
        except FileNotFoundError:
            df = pd.read_csv(
                op.join(self.portfolio_dir, f"{pf_name}_100.csv"), index_col=0
            )
        return df


def main():

    # 创建 DataFrame
    signal_df = pd.DataFrame(pd.read_csv('data/resources/NN1_best_predictions.csv'))
    signal_df['ending_date'] = pd.to_datetime(signal_df['ending_date'], errors='coerce')
    signal_df.set_index(["ending_date", "StockID"], inplace=True)

    # 假设 portfolio_dir 是保存结果的文件夹路径
    portfolio_dir = 'outputs'

    # 创建 PortfolioManager 实例
    portfolio_manager = PortfolioManager(
        signal_df=signal_df,
        freq="month",  # 设置为月度重平衡
        portfolio_dir=portfolio_dir,
        start_year=1987,  # 分析开始年份
        end_year=2016,  # 分析结束年份
        delay_list=[0],  # 设置延迟为 0
        load_signal=True,  # 加载信号数据
        transaction_cost=False  # 不考虑交易成本
    )

    # 计算投资组合的回报（等权重）
    portfolio_ret = portfolio_manager.generate_portfolio(
        weight_type="ew",  # 使用等权重
        cut=10,  # 将股票分为 5 个分位（deciles）
        delay=0  # 延迟为 0
    )

    # 输出投资组合的回报
    print(portfolio_ret)

    # 获取投资组合的回报总结
    portfolio_summary = portfolio_manager.load_portfolio_summary(weight_type="ew", cut=10)
    print(portfolio_summary)


if __name__ == "__main__":
    main()
