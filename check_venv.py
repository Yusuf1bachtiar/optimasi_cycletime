import sys
import pip

print(f"Python executable: {sys.executable}")
print("Installed packages:")
for package in pip.get_installed():
    print(f"- {package.key} ({package.version})")
