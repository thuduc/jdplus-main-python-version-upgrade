"""Easter date calculations."""

from datetime import date
from enum import Enum


class EasterAlgorithm(Enum):
    """Easter calculation algorithm."""
    
    GREGORIAN = "gregorian"  # Western Easter
    JULIAN = "julian"        # Orthodox Easter


def easter_date(year: int, algorithm: EasterAlgorithm = EasterAlgorithm.GREGORIAN) -> date:
    """Calculate Easter date for given year.
    
    Args:
        year: Year
        algorithm: Calculation algorithm
        
    Returns:
        Easter date
    """
    if algorithm == EasterAlgorithm.GREGORIAN:
        return _gregorian_easter(year)
    else:
        return _julian_easter(year)


def _gregorian_easter(year: int) -> date:
    """Calculate Gregorian (Western) Easter date.
    
    Uses the algorithm by Jean Meeus.
    
    Args:
        year: Year
        
    Returns:
        Easter date
    """
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    
    return date(year, month, day)


def _julian_easter(year: int) -> date:
    """Calculate Julian (Orthodox) Easter date.
    
    Uses the Meeus Julian algorithm.
    
    Args:
        year: Year
        
    Returns:
        Easter date (in Gregorian calendar)
    """
    a = year % 4
    b = year % 7
    c = year % 19
    d = (19 * c + 15) % 30
    e = (2 * a + 4 * b - d + 34) % 7
    
    month = (d + e + 114) // 31
    day = ((d + e + 114) % 31) + 1
    
    # Convert from Julian to Gregorian calendar
    # This is a simplified conversion
    julian_date = date(year, month, day)
    
    # Calculate difference between Julian and Gregorian calendars
    century = year // 100
    if year < 1582:
        offset = 0
    elif year < 1700:
        offset = 10
    elif year < 1800:
        offset = 11
    elif year < 1900:
        offset = 12
    elif year < 2100:
        offset = 13
    else:
        offset = 14
    
    # Add offset days
    from datetime import timedelta
    gregorian_date = julian_date + timedelta(days=offset)
    
    return gregorian_date


def easter_related_holidays(year: int, 
                          algorithm: EasterAlgorithm = EasterAlgorithm.GREGORIAN) -> dict:
    """Get common Easter-related holidays.
    
    Args:
        year: Year
        algorithm: Calculation algorithm
        
    Returns:
        Dictionary of holiday names to dates
    """
    from datetime import timedelta
    
    easter = easter_date(year, algorithm)
    
    holidays = {
        "Ash Wednesday": easter - timedelta(days=46),
        "Palm Sunday": easter - timedelta(days=7),
        "Maundy Thursday": easter - timedelta(days=3),
        "Good Friday": easter - timedelta(days=2),
        "Easter Sunday": easter,
        "Easter Monday": easter + timedelta(days=1),
        "Ascension": easter + timedelta(days=39),
        "Pentecost": easter + timedelta(days=49),
        "Whit Monday": easter + timedelta(days=50),
        "Trinity Sunday": easter + timedelta(days=56),
        "Corpus Christi": easter + timedelta(days=60),
    }
    
    return holidays