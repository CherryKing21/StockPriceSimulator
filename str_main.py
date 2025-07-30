import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import plotly.graph_objects as go
import pandas as pd
import base64


st.title("Everything is :blue[Random]")  
st.header("Stock Price Movement Simulator")


tabGBM, tabMC, tabDisc = st.tabs(["Geometric Brownian Motion Simulator", "Monte Carlo Simulator", "Note"])
with tabGBM:
    st.header(":green[Explanation]")
    st.write("""
Geometric Brownian Motion (:blue[GBM]) is a stochastic process. Brownian motion, initially observed in pollen particles, forms the basis for :blue[GBM], which is widely used in finance to model the random movement of stocks. This adopts the :green[weak form of the efficient market hypothesis (EMH)] and assumes stock prices follow a random walk with both a :purple[drift] and a :orange[volatility] component—meaning price changes are influenced by a predictable trend and random fluctuations. However, prices are independent of past information and rely only on the current state of the stock price (a :red[Markov process]).

Put simply, imagine a toddler starting from a particular point, hobbling either to the left or right depending on their balance.
""")


    st.latex(r"S_{t+1} = S_t \cdot \exp\left(\left(\mu - \frac{1}{2}\sigma^2\right)\Delta t + \sigma\sqrt{\Delta t} \cdot Z_t\right)")
    with st.expander("Variables Defined"):
        #latex stuff to define formula components
        st.latex(r"S_{t+1}")
        st.write("The stock price at the next time step (future price)")
        st.latex(r"S_t")
        st.write("The current stock price at time t")
        st.latex(r"\mu \text{ (mu)}")
        st.write("The drift rate or expected return - represents the average rate of growth of the stock price over time. For example, if μ = 0.08, the stock is expected to grow at 8% annually on average.")
        st.latex(r"\sigma \text{ (sigma)}")
        st.write("The volatility - measures how much the stock price fluctuates around its expected growth. Higher σ means more price variability and risk. For example, σ = 0.20 means 20% annual volatility.")
        st.latex(r"\Delta t \text{ (Delta t)}")
        st.write("The time increment - the small step forward in time. For daily simulations, Δt = 1/252 (since there are ~252 trading days per year). For monthly steps, Δt = 1/12.")
        st.latex(r"Z_t")
        st.write("A random normal variable - represents the unpredictable, random component of price movements. It's drawn from a standard normal distribution (mean = 0, standard deviation = 1).")
        st.markdown("#### Formula Components:")
        st.latex(r"(\mu - \frac{1}{2}\sigma^2)\Delta t")
        st.write("The deterministic drift component - this is the predictable growth adjusted for volatility")
        st.latex(r"\sigma\sqrt{\Delta t} \cdot Z_t")
        st.write("The stochastic (random) component - this adds the unpredictable price fluctuations")
        st.latex(r"\exp(...)")
        st.write("The exponential function ensures stock prices remain positive and creates the multiplicative (percentage-based) price changes that we observe in real markets")
        st.write("This formula captures both the expected upward trend of stock prices and their random daily fluctuations.")
    
    st.divider()
    
    ticker = st.text_input("Symbol Select","", 5, placeholder= "e.g. AAPL")

    #session state stuff for comfort
    if 'S_0' not in st.session_state: 
        st.session_state.S_0 = -1
    if 'super_hist' not in st.session_state:
        st.session_state.super_hist = None
    if 'metric' not in st.session_state:
        st.session_state.metric = None
    #button stuff

    def tickletest(ticker):
        tickle_test = yf.Ticker(ticker)
        tickle_hist = tickle_test.history(period = "1d")
        if tickle_hist.empty: 
            return False
        else:
            return True

    if st.button("Load Data", type = "secondary"):
        if not tickletest(ticker):
            st.error("This stock does not exist, please try again")
        else:
            main_ticker= yf.Ticker(ticker) 
            hist = main_ticker.history(period = "2d")
            st.session_state.S_0 = main_ticker.history(period = "1d")['Close'].iloc[-1]
            st.session_state.metric = {'price': st.session_state.S_0, 'change':(hist['Close'].iloc[-1]/hist['Close'].iloc[-2]-1)*100,'ticker': ticker}
            st.session_state.super_hist = main_ticker.history(period = "30d")['Close'].values
            # st.write(st.session_state.super_hist) // testing stuff
    if st.session_state.metric:
        st.metric("Latest Close", f"{st.session_state.metric['price']:+.2f} USD", f"{st.session_state.metric['change']:+.2f}%", "normal")

    
    st.divider()

    if st.session_state.S_0 == -1:
        st.badge("S₀ not defined yet", color='red')
    else: 
        col1, col2 = st.columns([2,9])
        with col1:
            st.badge(f"S₀ = {st.session_state.S_0:.2f}", color = 'orange')
        with col2:
            st.badge("Δt = 1/252 (1 year / 252 trading days)", color = 'orange')

    mew = st.number_input("Expected Return (μ)", min_value=0.00, max_value=1.00, value=0.1, step= 0.01, placeholder="decimal percentage")
    sig = st.number_input("Annual Vol (σ)", min_value=0.00, max_value=1.00, value= 0.113, step = 0.01, placeholder="decimal percentage")
    t_years = 1
    n_steps = 252
    delta_t = t_years/n_steps
    Z = np.random.normal(0, 1, n_steps)
    S = np.zeros(n_steps)
    S[0] = st.session_state.S_0

    for i in range(1,n_steps):
        S[i] = S[i-1] * np.exp((mew - 0.5 * sig**2)*delta_t + sig*np.sqrt(delta_t)*Z[i])
        #st.write(st.session_state.super_hist)
        #st.write(S)
    st.divider()    

    coli, colj = st.columns([4,3])
    with coli:
        st.header(":green[Plot]")
    with colj:
        if st.session_state.metric:
            st.metric("Latest Close", f"{st.session_state.metric['price']:+.2f} USD", f"{st.session_state.metric['change']:+.2f}%", "normal")
    if st.session_state.super_hist is None:
        st.error("Choose a stock first")
    else:  
        data_burger = np.concatenate([st.session_state.super_hist, S]) 
        fig = go.Figure()

        
        fig.add_trace(go.Scatter(x=list(range(len(st.session_state.super_hist))), 
        y=st.session_state.super_hist, 
        mode='lines', 
        name='Historical Price',
        line=dict(color='green')
        ))

        fig.add_trace(go.Scatter(x=list(range(len(st.session_state.super_hist), len(data_burger))), 
        y=S, 
        mode='lines', 
        name='Simulated Price',
        line=dict(color='yellow')
        ))

        fig.update_layout(
        title=f"{st.session_state.metric['ticker']} Historical + GBM Simulation",
        xaxis_title="Days",
        yaxis_title="Price ($)",
        hovermode='x'
        )
        st.plotly_chart(fig, use_container_width=True)

with tabMC:
    st.header(":green[Explanation]")
    st.write("""
The :blue[Monte Carlo simulation] uses GBM to model and simulate potential future price movements of stocks. By simulating thousands of price paths based on GBM, one can estimate the probability distribution of future asset prices and assess potential :red[risks] and :green[returns].

Continuing our toddler analogy, this is like letting an army of toddlers hobble out into a park from a common starting point and analyzing where most of them end up.

In the end, we produce a histogram of returns (percentage and absolute on a $1M USD portfolio). Two important observations:

The returns are :violet[lognormally distributed], meaning there is a right skew to their distribution. This happens because a stock price can never go below zero but can (technically) increase indefinitely.

The element of :violet[drift] is still observed via the mean line, despite the random shocks. With a sufficient number of simulations, the average outcome tends to converge toward the drift.

We also calculate a :red[Value at Risk (VaR)] at a 95% confidence level—this represents the loss your portfolio is expected to incur 5% of the time. In toddler terms, it’s the bottom 5% that ran into the pond.

The simulation runs 1,000 random walks on a 3-stock portfolio with customizable features, taking :blue[correlations] into account.

:blue[Correlation] is important because we want to model the performance of a portfolio where assets can behave similarly or oppositely, which alters the :red[risk profile] of the portfolio.

If all the toddlers love ice cream and there’s an ice cream stand by the pond, they’ll all end up there. But if the toddlers have mixed preferences, that event will have less impact on the overall distribution of toddlers (i.e., reduced risk).
""")
    st. divider()

    def get_corr_matrix(tickers, period="1y"):
        data = yf.download(tickers, start=None, period=period)['Close']
        returns = data.pct_change().dropna()
        correlation_matrix = returns.corr()
        return correlation_matrix

    if 'portfolio_loaded' not in st.session_state:
        st.session_state.portfolio_loaded = False
    if 'portfolio_data' not in st.session_state:
        st.session_state.portfolio_data = None

    col1, col2, col3 = st.columns([3,3,3])
    with col1:
        ticker1 = st.text_input("Symbol 1","", 5, placeholder= "e.g. AAPL")
    with col2:
        ticker2 = st.text_input("Symbol 2","", 5, placeholder= "e.g. NVDA")
    with col3:
        ticker3 = st.text_input("Symbol 3","", 5, placeholder= "e.g. PLTR")
    
    def tickletest_portfolio(ticker1, ticker2, ticker3):
        tickers = [ticker1, ticker2, ticker3]
        for t in tickers:
            if not t or t.strip() == "":
                return False
            
        for i in tickers:
            try: 
                test_i = yf.Ticker(i.strip().upper())
                hist_i = test_i.history(period = "1d")
                if hist_i.empty:
                    return False
                
            except: 
                return False
        return True
    
    if st.button("Load Portfolio", disabled = not tickletest_portfolio(ticker1, ticker2, ticker3), type = "primary"):
        tickers = [ticker1, ticker2, ticker3]
        #st.write(tickers)

        def getPrice(t):
            tic = yf.Ticker(t)
            close = tic.history(period = "1d")
            return close['Close'].iloc[-1]
        
        S_03 = np.array([getPrice(i) for i in tickers])
        #st.write(S_03)
        st.session_state.portfolio_loaded = True
        st.session_state.portfolio_data = {'tickers': tickers,'prices': S_03}
    
    if st.session_state.portfolio_loaded:
    
        S_03 = st.session_state.portfolio_data['prices']  
    
        with st.expander("Customize", False):
            col1, col2, col3 = st.columns([3,3,3])
            with col1:
                w1 = st.number_input("Weight 1", 0.0, 1.0, 0.4, step = 0.1, placeholder= "e.g. 0.4")
                m1 = st.number_input("Expected Return 1", 0.0, 1.0, 0.1, step = 0.01, placeholder= "e.g. 0.1")
                s1 = st.number_input("Annual Vol 1", 0.0, 1.0, 0.1, step = 0.01, placeholder= "e.g. 0.1")
            with col2:
                w2 = st.number_input("Weight 2", 0.0, 1.0, 0.3, step = 0.1, placeholder= "e.g. 0.3")
                m2 = st.number_input("Expected Return 2", 0.0, 1.0, 0.12, step = 0.01, placeholder= "e.g. 0.12")
                s2 = st.number_input("Annual Vol 2", 0.0, 1.0, 0.1, step = 0.01, placeholder= "e.g. 0.1")
            with col3:
                w3 = st.number_input("Weight 3", 0.0, 1.0,0.3, step = 0.1, placeholder= "e.g. 0.3")
                m3 = st.number_input("Expected Return 3", 0.0, 1.0, 0.13, step = 0.01, placeholder= "e.g. 0.15")
                s3 = st.number_input("Annual Vol 3", 0.0, 1.0, 0.1, step = 0.01, placeholder= "e.g. 0.1")
            
            if w1+w2+w3 != 1:
                st.error("Weights need to add up to 1") 

        if st.button("Rerun Simulation", type="secondary"):
            weights = np.array([w1,w2,w3])
            mews = np.array([m1,m2,m3])
            sigs = np.array([s1,s2,s3])
            #st.write(weights)
            #st.write(mews)
            #st.write(sigs)
        

            corr_matrix = get_corr_matrix(st.session_state.portfolio_data['tickers'])
            #st.write(corr_matrix)
            cov_matrix = np.outer(sigs, sigs) * corr_matrix
            #st.write(cov_matrix)
            port_size = 3
            t_mc = 1
            n_mc = 252
            dt_mc = t_mc/n_mc
            L = np.linalg.cholesky(cov_matrix)
            #st.write(L)
            sims = 1000

            price_arrs = np.zeros((sims, n_mc+1,port_size))
            price_arrs[:, 0, :] = S_03

            for sim in range(sims):
                shock = np.random.normal(size=(n_mc, port_size))
                correlated_shock = shock @ L.T
                for t in range(1, n_mc + 1):
                    price_arrs[sim, t, :] = price_arrs[sim, t - 1, :] * np.exp(
                        (mews - 0.5 * sigs**2) * dt_mc + np.sqrt(dt_mc) * correlated_shock[t - 1]
                    )
            
            portfolio_vals = np.sum(price_arrs * weights, axis = 2)
            #st.write(portfolio_vals)

            portfolio_value_0 = np.sum(S_03 * weights)
            scale = 1000000 / portfolio_value_0
            portfolio_vals *= scale


            

            
            df_summary = pd.DataFrame({
                'Stock': st.session_state.portfolio_data['tickers'],
                'Weight': weights,
                'Starting Price ($)': S_03,
                'Portfolio Value ($)': weights * S_03
            })

            st.subheader("Portfolio Composition")
            st.dataframe(
                df_summary.style.format({
                    'Weight': '{:.1%}',
                    'Starting Price ($)': '${:.2f}',
                    'Portfolio Value ($)': '${:.2f}'
                }),
                use_container_width=True
            )

        
            # Plotting MC
            fig = go.Figure()
            
            for i in range(min(100, sims)):
                fig.add_trace(go.Scatter(
                    y=portfolio_vals[i], 
                    mode='lines', 
                    line=dict(width=0.5, color='rgba(50,150,200,0.4)'),
                    showlegend=False,
                    hovertemplate='Day: %{x}<br>Portfolio Value: $%{y:.2f}<extra></extra>'
                ))
            
            mean_path = np.mean(portfolio_vals, axis=0)
            fig.add_trace(go.Scatter(
                y=mean_path,
                mode='lines',
                name='Mean Path',
                line=dict(width=3, color='red')
            ))
                
            fig.update_layout(
                title="Portfolio Monte Carlo Simulation (100 paths shown)",
                xaxis_title="Days",
                yaxis_title="Portfolio Value ($)",
                hovermode='x'
            )
            
            st.plotly_chart(fig, use_container_width=True)

            initial_portfolio_value = portfolio_vals[0, 0]
            final_portfolio_values = portfolio_vals[:, -1]
            portfolio_returns = (final_portfolio_values - initial_portfolio_value) / initial_portfolio_value * 100

            #histogram
            fig_hist = go.Figure()

            fig_hist.add_trace(go.Histogram(
                x=portfolio_returns,
                nbinsx=50,
                name='Portfolio Returns',
                marker=dict(color='steelblue', opacity=0.7),
                hovertemplate='Return: %{x:.2f}%<br>Count: %{y}<extra></extra>'
            ))

        
            mean_return = np.mean(portfolio_returns)
            fig_hist.add_vline(
                x=mean_return, 
                line_dash="dash", 
                line_color="red",
                annotation_text=f"Mean: {mean_return:.2f}%", 
                annotation_position = "top"
            )



            fig_hist.add_vline(
                x=0,
                line_dash="dot",
                line_color="lightgreen",
                annotation_text=f"Starting Value", 
                annotation_position = "bottom left"
            )

            fig_hist.update_layout(
                title="Distribution of Portfolio Returns (1 Year)",
                xaxis_title="Return (%)",
                yaxis_title="Frequency",
                showlegend=False
            )

            st.plotly_chart(fig_hist, use_container_width=True)

            #hist actual
            fig_hist2 = go.Figure()

            fig_hist2.add_trace(go.Histogram(
                x=final_portfolio_values,
                nbinsx=50,
                name='Portfolio Values',
                marker=dict(color='darkgreen', opacity=0.7),
                hovertemplate='Portfolio Value: $%{x:,.0f}<br>Count: %{y}<extra></extra>'
            ))

            
            mean_final_value = np.mean(final_portfolio_values)
            fig_hist2.add_vline(
                x=mean_final_value, 
                line_dash="dash", 
                line_color="red",
                annotation_text=f"Mean: ${mean_final_value:,.0f}", 
                annotation_position="top"
            )

            
            var_5_value = np.percentile(final_portfolio_values, 5)
            fig_hist2.add_vline(
                x=var_5_value,
                line_dash="dash",
                line_color="orange", 
                annotation_text=f"5% VaR: ${var_5_value:,.0f}",
                annotation_position="top left"
            )

            
            fig_hist2.add_vline(
                x=1000000,
                line_dash="dot",
                line_color="lightblue",
                annotation_text=f"Starting Value", 
                annotation_position="bottom left"
            )

            fig_hist2.update_layout(
                title="Distribution of Final Portfolio Values (1 Year)",
                xaxis_title="Portfolio Value ($)",
                yaxis_title="Frequency",
                showlegend=False
            )

            st.plotly_chart(fig_hist2, use_container_width=True)
        else:
            st.info("Click 'Rerun Simulation' to generate new results with your parameters")


    else:
        st.error("Please fill in all tickers")

with tabDisc:
    st.header("Note and :red[Disclaimer]")
    st.write("""
My name is Lev, and I’m a business/finance major passionate about :green[data], :blue[technology], and :violet[quantitative methods in finance]. I hope this tool helps you learn as much as I did making it! All the code is available in the public GitHub repo linked on the page (click the little GitHub cat icon).

Keep in mind that this tool should not be used for investment decisions—it’s intended for :red[educational purposes] only.

Feel free to reach out to me on LinkedIn with any questions or suggestions!
             
Stylistic formatting and debugging were assisted by Claude Sonnet 4.
""")

    
    col1, col2 = st.columns([3,4])
    with col1:
        st.image("IMG_0781.jpg", width = 300)
    with col2:
        st.subheader("Lev Akhmerov, 21")
        #st.link_button("LinkedIn", "https://www.linkedin.com/in/lev-akhmerov/")
        #st.link_button("Gmail", "akhmerovlev@gmail.com")
        #st.link_button("📧 email", "mailto:akhmerovlev@gmail.com")

       
        col1, col2, col3 = st.columns([1, 1, 5])

        with col1:
            with open(r"InBug-White.png", "rb") as file:
                contents = file.read()
                data_url = base64.b64encode(contents).decode("utf-8")
            
            st.markdown(f"""
            <a href="https://www.linkedin.com/in/lev-akhmerov/" target="_blank">
                <img src="data:image/png;base64,{data_url}" width="40" height="40" style="cursor: pointer;">
            </a>
            """, unsafe_allow_html=True)

        with col2:
            with open(r"e-mail-mail-letter-white-logo-icon-transparent-background-701751694973962w7ooykuds3.png", "rb") as file:
                contents = file.read()
                data_url = base64.b64encode(contents).decode("utf-8")
            
            st.markdown(f"""
            <a href="mailto:akhmerovlev@gmail.com" target="_blank">
                <img src="data:image/png;base64,{data_url}" width="40" height="40" style="cursor: pointer;">
            </a>
            """, unsafe_allow_html=True)

  
    

    
    
     







    
