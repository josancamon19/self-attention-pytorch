# Creates a zip file for submission on Gradescope.

import os
import zipfile

required_files = [p for p in os.listdir('.') if p.endswith('.py')] + \
                 [f'predictions/{p}' for p in os.listdir('predictions')] + \
                     [f'models/{p}' for p in os.listdir('models')] + \
                        [f'modules/{p}' for p in os.listdir('modules')]

def main():
    aid = 'cs224n_default_final_project_submission'

    with zipfile.ZipFile(f"{aid}.zip", 'w') as zz:
        for file in required_files:
            zz.write(file, os.path.join(".", file))
    print(f"Submission zip file created: {aid}.zip")

if __name__ == '__main__':
    main()
