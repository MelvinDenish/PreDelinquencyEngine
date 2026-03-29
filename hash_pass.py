from scoring_service.auth import get_password_hash
import sys
password = sys.argv[1]
print(get_password_hash(password))
