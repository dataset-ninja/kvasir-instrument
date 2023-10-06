from typing import Dict, List, Optional, Union

from dataset_tools.templates import (
    AnnotationType,
    Category,
    CVTask,
    Domain,
    Industry,
    License,
    Research,
)

##################################
# * Before uploading to instance #
##################################
PROJECT_NAME: str = "Kvasir Instrument"
PROJECT_NAME_FULL: str = "Kvasir Instrument: A Gastrointestinal Instrument Dataset"
HIDE_DATASET = False  # set False when 100% sure about repo quality

##################################
# * After uploading to instance ##
##################################
LICENSE: License = License.Custom(url="https://datasets.simula.no/kvasir-instrument/")
APPLICATIONS: List[Union[Industry, Domain, Research]] = [Industry.Medical()]
CATEGORY: Category = Category.Medical(extra=Category.Robotics(), sensitive_content=True)

CV_TASKS: List[CVTask] = [CVTask.InstanceSegmentation(), CVTask.SemanticSegmentation(), CVTask.ObjectDetection()]
ANNOTATION_TYPES: List[AnnotationType] = [AnnotationType.InstanceSegmentation(), AnnotationType.ObjectDetection()]

RELEASE_DATE: Optional[str] = "2020-08-14"  # e.g. "YYYY-MM-DD"
if RELEASE_DATE is None:
    RELEASE_YEAR: int = None

HOMEPAGE_URL: str = "https://datasets.simula.no/kvasir-instrument/"
# e.g. "https://some.com/dataset/homepage"

PREVIEW_IMAGE_ID: int = 6765217
# This should be filled AFTER uploading images to instance, just ID of any image.

GITHUB_URL: str = "https://github.com/dataset-ninja/kvasir-instrument"
# URL to GitHub repo on dataset ninja (e.g. "https://github.com/dataset-ninja/some-dataset")

##################################
### * Optional after uploading ###
##################################
DOWNLOAD_ORIGINAL_URL: Optional[Union[str, dict]] = "https://datasets.simula.no/downloads/kvasir-instrument.zip"
# Optional link for downloading original dataset (e.g. "https://some.com/dataset/download")

CLASS2COLOR: Optional[Dict[str, List[str]]] = None
# If specific colors for classes are needed, fill this dict (e.g. {"class1": [255, 0, 0], "class2": [0, 255, 0]})

# If you have more than the one paper, put the most relatable link as the first element of the list
# Use dict key to specify name for a button
PAPER: Optional[Union[str, List[str], Dict[str, str]]] = "https://arxiv.org/abs/2011.08065v1"
BLOGPOST: Optional[Union[str, List[str], Dict[str, str]]] = None
REPOSITORY: Optional[Union[str, List[str], Dict[str, str]]] = {"Repository":"https://github.com/DebeshJha/Kvasir-Instrument", "OSF Repository": "https://osf.io/kp6my"}

CITATION_URL: Optional[str] = "https://datasets.simula.no/kvasir-instrument/"
AUTHORS: Optional[List[str]] = ["Debesh Jha", "Sharib Ali", "Krister Emanuelsen", "Steven A. Hicks", "Vajira Thambawita", "Enrique Garcia-Ceja", "Michael A. Riegler", "Thomas de Lange", "Peter T. Schmidt", "Håvard D. Johansen", "Dag Johansen" , "Pål Halvorsen"]

ORGANIZATION_NAME: Optional[Union[str, List[str]]] = "NOR-SWE-UK joint research group"
ORGANIZATION_URL: Optional[Union[str, List[str]]] = "https://link.springer.com/chapter/10.1007/978-3-030-67835-7_19"

# Set '__PRETEXT__' or '__POSTTEXT__' as a key with string value to add custom text. e.g. SLYTAGSPLIT = {'__POSTTEXT__':'some text}
SLYTAGSPLIT: Optional[Dict[str, Union[List[str], str]]] = None
TAGS: Optional[List[str]] = None


SECTION_EXPLORE_CUSTOM_DATASETS: Optional[List[str]] = None

##################################
###### ? Checks. Do not edit #####
##################################


def check_names():
    fields_before_upload = [PROJECT_NAME]  # PROJECT_NAME_FULL
    if any([field is None for field in fields_before_upload]):
        raise ValueError("Please fill all fields in settings.py before uploading to instance.")


def get_settings():
    if RELEASE_DATE is not None:
        global RELEASE_YEAR
        RELEASE_YEAR = int(RELEASE_DATE.split("-")[0])

    settings = {
        "project_name": PROJECT_NAME,
        "project_name_full": PROJECT_NAME_FULL or PROJECT_NAME,
        "hide_dataset": HIDE_DATASET,
        "license": LICENSE,
        "applications": APPLICATIONS,
        "category": CATEGORY,
        "cv_tasks": CV_TASKS,
        "annotation_types": ANNOTATION_TYPES,
        "release_year": RELEASE_YEAR,
        "homepage_url": HOMEPAGE_URL,
        "preview_image_id": PREVIEW_IMAGE_ID,
        "github_url": GITHUB_URL,
    }

    if any([field is None for field in settings.values()]):
        raise ValueError("Please fill all fields in settings.py after uploading to instance.")

    settings["release_date"] = RELEASE_DATE
    settings["download_original_url"] = DOWNLOAD_ORIGINAL_URL
    settings["class2color"] = CLASS2COLOR
    settings["paper"] = PAPER
    settings["blog"] = BLOGPOST
    settings["repository"] = REPOSITORY
    settings["citation_url"] = CITATION_URL
    settings["authors"] = AUTHORS
    settings["organization_name"] = ORGANIZATION_NAME
    settings["organization_url"] = ORGANIZATION_URL
    settings["slytagsplit"] = SLYTAGSPLIT
    settings["tags"] = TAGS

    settings["explore_datasets"] = SECTION_EXPLORE_CUSTOM_DATASETS

    return settings
