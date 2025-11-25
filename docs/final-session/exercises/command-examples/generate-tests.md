# /generate-tests: Automatische Unit Test Generierung

Analysiere die bereitgestellte Funktion oder Klasse und generiere umfassende Unit Tests.

## Aufgaben:
1. **Code-Struktur analysieren**
   - Funktions-/Klassen-Parameter identifizieren
   - Return-Types und mögliche Exceptions verstehen
   - Dependencies und externe Calls erkennen

2. **Test-Szenarien entwickeln**
   - Happy Path Tests (normale Verwendung)
   - Edge Cases (Grenzwerte, leere Inputs)
   - Error Cases (ungültige Parameter, Exceptions)
   - Integration Tests (falls externe Dependencies)

3. **Test-Code generieren**
   - Passende Test-Framework Syntax verwenden
   - Setup/Teardown Methoden erstellen
   - Mocking für externe Dependencies
   - Assertions für alle Return Values und Side Effects

## Output Format:

### Python (pytest):
```python
import pytest
from unittest.mock import Mock, patch
from your_module import your_function

class TestYourFunction:
    def setup_method(self):
        """Setup vor jedem Test"""
        pass
    
    def test_happy_path(self):
        """Test normale Verwendung"""
        result = your_function(valid_input)
        assert result == expected_output
    
    def test_edge_cases(self):
        """Test Grenzwerte"""
        assert your_function("") == ""
        assert your_function(None) is None
    
    def test_error_handling(self):
        """Test Exception-Handling"""
        with pytest.raises(ValueError):
            your_function(invalid_input)
    
    @patch('your_module.external_dependency')
    def test_with_mocking(self, mock_dependency):
        """Test mit gemockten Dependencies"""
        mock_dependency.return_value = "mocked_result"
        result = your_function("input")
        assert result == "expected_with_mock"
```

### JavaScript (Jest):
```javascript
const { yourFunction } = require('./your-module');

describe('yourFunction', () => {
    beforeEach(() => {
        // Setup vor jedem Test
    });

    test('should handle normal input correctly', () => {
        const result = yourFunction('valid input');
        expect(result).toBe('expected output');
    });

    test('should handle edge cases', () => {
        expect(yourFunction('')).toBe('');
        expect(yourFunction(null)).toBeNull();
    });

    test('should throw error for invalid input', () => {
        expect(() => {
            yourFunction('invalid');
        }).toThrow('Expected error message');
    });

    test('should work with mocked dependencies', () => {
        const mockDependency = jest.fn().mockReturnValue('mocked');
        // Test implementation
    });
});
```

## Test-Coverage Ziele:
- **Statements:** 90%+
- **Branches:** 85%+
- **Functions:** 100%
- **Lines:** 90%+

## Beispiel-Anwendung:

### Input:
```python
def validate_password(password: str) -> dict:
    """Validiert ein Passwort nach Sicherheitskriterien"""
    if not password:
        raise ValueError("Password cannot be empty")
    
    result = {
        'valid': True,
        'errors': []
    }
    
    if len(password) < 8:
        result['valid'] = False
        result['errors'].append("Password must be at least 8 characters")
    
    if not any(c.isupper() for c in password):
        result['valid'] = False
        result['errors'].append("Password must contain uppercase letter")
    
    return result
```

### Output:
```python
import pytest
from password_validator import validate_password

class TestValidatePassword:
    def test_valid_password(self):
        """Test mit gültigem Passwort"""
        result = validate_password("SecurePass123!")
        assert result['valid'] is True
        assert result['errors'] == []
    
    def test_empty_password_raises_error(self):
        """Test leeres Passwort wirft ValueError"""
        with pytest.raises(ValueError, match="Password cannot be empty"):
            validate_password("")
    
    def test_none_password_raises_error(self):
        """Test None Passwort wirft ValueError"""
        with pytest.raises(ValueError):
            validate_password(None)
    
    def test_too_short_password(self):
        """Test zu kurzes Passwort"""
        result = validate_password("Short1!")
        assert result['valid'] is False
        assert "Password must be at least 8 characters" in result['errors']
    
    def test_no_uppercase_letter(self):
        """Test Passwort ohne Grossbuchstaben"""
        result = validate_password("lowercase123!")
        assert result['valid'] is False
        assert "Password must contain uppercase letter" in result['errors']
    
    def test_multiple_validation_errors(self):
        """Test Passwort mit mehreren Fehlern"""
        result = validate_password("short")
        assert result['valid'] is False
        assert len(result['errors']) == 2
        assert "Password must be at least 8 characters" in result['errors']
        assert "Password must contain uppercase letter" in result['errors']
    
    def test_minimum_valid_password(self):
        """Test minimal gültiges Passwort"""
        result = validate_password("Password")
        assert result['valid'] is True
        assert result['errors'] == []
```

## Best Practices:
- **Descriptive Test Names:** Was wird getestet?
- **AAA Pattern:** Arrange, Act, Assert
- **One Assertion per Test:** Fokussiert und debuggbar
- **Test Data Builders:** Für komplexe Test-Objekte
- **Parameterized Tests:** Für ähnliche Test-Cases
- **Mocking Strategy:** Nur externe Dependencies mocken
