import os

def list_pth_folders():
    base_path = '/data/ASE/output'
    pth_folders = []
    
    # Walk through all subdirectories
    for subfolder in sorted(os.listdir(base_path)):
        nn_path = os.path.join(base_path, subfolder, 'nn')
        
        # Check if nn folder exists and contains .pth files
        if os.path.isdir(nn_path):
            pth_files = [f for f in os.listdir(nn_path) if f.endswith('.pth')]
            if pth_files:
                print(subfolder)

if __name__ == '__main__':
    list_pth_folders()
