import numpy as np
from scipy.stats import norm
from typing import Dict, Tuple, Optional
import warnings

class AdvancedBlackScholes:
    """
    Advanced Black-Scholes Option Pricing Model with comprehensive Greeks calculation
    and enhanced validation features.
    
    This class implements the Black-Scholes-Merton model for European option pricing
    with additional features for risk management and sensitivity analysis.
    """
    
    def __init__(
        self,
        spot_price: float,
        strike_price: float,
        time_to_expiry: float,
        volatility: float,
        risk_free_rate: float,
        dividend_yield: float = 0.0
    ):
        """
        Initialize the Black-Scholes model with market parameters.
        
        Args:
            spot_price (float): Current price of the underlying asset
            strike_price (float): Strike price of the option
            time_to_expiry (float): Time to expiration in years
            volatility (float): Annualized volatility (as a decimal, e.g., 0.2 for 20%)
            risk_free_rate (float): Risk-free interest rate (as a decimal)
            dividend_yield (float): Continuous dividend yield (default: 0.0)
        
        Raises:
            ValueError: If any parameter is invalid
        """
        self._validate_inputs(spot_price, strike_price, time_to_expiry, volatility, risk_free_rate, dividend_yield)
        
        self.S = spot_price
        self.K = strike_price
        self.T = time_to_expiry
        self.sigma = volatility
        self.r = risk_free_rate
        self.q = dividend_yield
        
        # Initialize calculated values
        self._d1 = None
        self._d2 = None
        self._prices_calculated = False
        self._greeks_calculated = False
        
    def _validate_inputs(self, S: float, K: float, T: float, sigma: float, r: float, q: float) -> None:
        """Validate input parameters for the Black-Scholes model."""
        if S <= 0:
            raise ValueError("Spot price must be positive")
        if K <= 0:
            raise ValueError("Strike price must be positive")
        if T < 0:
            raise ValueError("Time to expiry cannot be negative")
        if T == 0:
            warnings.warn("Time to expiry is zero - option has expired")
        if sigma < 0:
            raise ValueError("Volatility cannot be negative")
        if sigma > 5:
            warnings.warn(f"Volatility of {sigma:.1%} seems unusually high")
        if abs(r) > 1:
            warnings.warn(f"Risk-free rate of {r:.1%} seems unusual")
        if abs(q) > 1:
            warnings.warn(f"Dividend yield of {q:.1%} seems unusual")
    
    def _calculate_d_parameters(self) -> Tuple[float, float]:
        """Calculate d1 and d2 parameters used in Black-Scholes formula."""
        if self.T == 0:
            # Handle expiration case
            if self.S > self.K:
                return float('inf'), float('inf')
            elif self.S < self.K:
                return float('-inf'), float('-inf')
            else:
                return 0, 0
        
        d1 = (np.log(self.S / self.K) + (self.r - self.q + 0.5 * self.sigma**2) * self.T) / (self.sigma * np.sqrt(self.T))
        d2 = d1 - self.sigma * np.sqrt(self.T)
        
        self._d1, self._d2 = d1, d2
        return d1, d2
    
    def calculate_prices(self) -> Dict[str, float]:
        """
        Calculate call and put option prices using Black-Scholes formula.
        
        Returns:
            Dict containing 'call' and 'put' option prices
        """
        d1, d2 = self._calculate_d_parameters()
        
        if self.T == 0:
            # At expiration
            call_price = max(self.S - self.K, 0)
            put_price = max(self.K - self.S, 0)
        else:
            # Standard Black-Scholes calculation
            call_price = (self.S * np.exp(-self.q * self.T) * norm.cdf(d1) - 
                         self.K * np.exp(-self.r * self.T) * norm.cdf(d2))
            put_price = (self.K * np.exp(-self.r * self.T) * norm.cdf(-d2) - 
                        self.S * np.exp(-self.q * self.T) * norm.cdf(-d1))
        
        self.call_price = call_price
        self.put_price = put_price
        self._prices_calculated = True
        
        return {'call': call_price, 'put': put_price}
    
    def calculate_greeks(self) -> Dict[str, Dict[str, float]]:
        """
        Calculate all Greeks (Delta, Gamma, Theta, Vega, Rho) for both call and put options.
        
        Returns:
            Dict containing Greeks for both call and put options
        """
        if not self._prices_calculated:
            self.calculate_prices()
        
        d1, d2 = self._d1, self._d2
        
        if self.T == 0:
            # At expiration, most Greeks are zero or undefined
            greeks = {
                'call': {'delta': 1 if self.S > self.K else 0, 'gamma': 0, 'theta': 0, 'vega': 0, 'rho': 0},
                'put': {'delta': -1 if self.S < self.K else 0, 'gamma': 0, 'theta': 0, 'vega': 0, 'rho': 0}
            }
        else:
            # Calculate standard Greeks
            # Delta
            call_delta = np.exp(-self.q * self.T) * norm.cdf(d1)
            put_delta = -np.exp(-self.q * self.T) * norm.cdf(-d1)
            
            # Gamma (same for call and put)
            gamma = (np.exp(-self.q * self.T) * norm.pdf(d1)) / (self.S * self.sigma * np.sqrt(self.T))
            
            # Theta
            theta_common = (-self.S * norm.pdf(d1) * self.sigma * np.exp(-self.q * self.T) / (2 * np.sqrt(self.T)) - 
                           self.r * self.K * np.exp(-self.r * self.T))
            call_theta = (theta_common * norm.cdf(d2) + self.q * self.S * norm.cdf(d1) * np.exp(-self.q * self.T)) / 365
            put_theta = (theta_common * norm.cdf(-d2) - self.q * self.S * norm.cdf(-d1) * np.exp(-self.q * self.T)) / 365
            
            # Vega (same for call and put)
            vega = self.S * np.exp(-self.q * self.T) * norm.pdf(d1) * np.sqrt(self.T) / 100
            
            # Rho
            call_rho = self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(d2) / 100
            put_rho = -self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(-d2) / 100
            
            greeks = {
                'call': {
                    'delta': call_delta,
                    'gamma': gamma,
                    'theta': call_theta,
                    'vega': vega,
                    'rho': call_rho
                },
                'put': {
                    'delta': put_delta,
                    'gamma': gamma,
                    'theta': put_theta,
                    'vega': vega,
                    'rho': put_rho
                }
            }
        
        self.greeks = greeks
        self._greeks_calculated = True
        return greeks
    
    def implied_volatility(self, market_price: float, option_type: str = 'call', 
                          max_iterations: int = 100, tolerance: float = 1e-6) -> Optional[float]:
        """
        Calculate implied volatility using Newton-Raphson method.
        
        Args:
            market_price (float): Market price of the option
            option_type (str): 'call' or 'put'
            max_iterations (int): Maximum number of iterations
            tolerance (float): Convergence tolerance
            
        Returns:
            Implied volatility or None if convergence fails
        """
        if option_type not in ['call', 'put']:
            raise ValueError("option_type must be 'call' or 'put'")
        
        # Initial guess
        iv = 0.2
        
        for i in range(max_iterations):
            # Create temporary model with current volatility guess
            temp_model = AdvancedBlackScholes(self.S, self.K, self.T, iv, self.r, self.q)
            prices = temp_model.calculate_prices()
            greeks = temp_model.calculate_greeks()
            
            price_diff = prices[option_type] - market_price
            vega = greeks[option_type]['vega'] * 100  # Convert back to full scale
            
            if abs(price_diff) < tolerance:
                return iv
            
            if vega == 0:
                return None
                
            # Newton-Raphson update
            iv = iv - price_diff / vega
            
            # Keep volatility positive
            iv = max(iv, 0.001)
        
        return None  # Failed to converge
    
    def get_summary(self) -> Dict:
        """
        Get a comprehensive summary of the option pricing analysis.
        
        Returns:
            Dict containing all calculated values and parameters
        """
        if not self._prices_calculated:
            self.calculate_prices()
        if not self._greeks_calculated:
            self.calculate_greeks()
        
        return {
            'parameters': {
                'spot_price': self.S,
                'strike_price': self.K,
                'time_to_expiry': self.T,
                'volatility': self.sigma,
                'risk_free_rate': self.r,
                'dividend_yield': self.q
            },
            'prices': {
                'call': self.call_price,
                'put': self.put_price
            },
            'greeks': self.greeks,
            'moneyness': self.S / self.K,
            'intrinsic_value': {
                'call': max(self.S - self.K, 0),
                'put': max(self.K - self.S, 0)
            },
            'time_value': {
                'call': self.call_price - max(self.S - self.K, 0),
                'put': self.put_price - max(self.K - self.S, 0)
            }
        }


# Example usage and testing
if __name__ == "__main__":
    # Example parameters
    spot_price = 100
    strike_price = 105
    time_to_expiry = 0.25  # 3 months
    volatility = 0.20      # 20%
    risk_free_rate = 0.05  # 5%
    
    # Create Black-Scholes model
    bs = AdvancedBlackScholes(
        spot_price=spot_price,
        strike_price=strike_price,
        time_to_expiry=time_to_expiry,
        volatility=volatility,
        risk_free_rate=risk_free_rate
    )
    
    # Calculate prices and Greeks
    prices = bs.calculate_prices()
    greeks = bs.calculate_greeks()
    
    print("=== Black-Scholes Option Pricing Results ===")
    print(f"Call Price: ${prices['call']:.4f}")
    print(f"Put Price: ${prices['put']:.4f}")
    print("\n=== Greeks ===")
    for option_type in ['call', 'put']:
        print(f"{option_type.upper()} Greeks:")
        for greek, value in greeks[option_type].items():
            print(f"  {greek.capitalize()}: {value:.6f}")
        print()
    
    # Test implied volatility
    market_call_price = prices['call']
    implied_vol = bs.implied_volatility(market_call_price, 'call')
    print(f"Implied Volatility (should match input): {implied_vol:.4f}")
    
    # Get complete summary
    summary = bs.get_summary()
    print(f"\nMoneyness (S/K): {summary['moneyness']:.4f}")
    print(f"Call Time Value: ${summary['time_value']['call']:.4f}")
    print(f"Put Time Value: ${summary['time_value']['put']:.4f}")