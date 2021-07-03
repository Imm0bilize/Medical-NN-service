import argparse


parser = argparse.ArgumentParser(
    description="Service for covid-19 segmentation and detection in dicom files",
    add_help=True, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--host", default='localhost')
parser.add_argument("--port", default=8080)
parser.add_argument("--path_to_log_dir", default='log/')
parser.add_argument("--time_to_shutdown_session", default=300.0,
                    help='Time in seconds after how long the model will be unloaded from RAM if it is not accessed')
args = parser.parse_args()