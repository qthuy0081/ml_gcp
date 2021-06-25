import pandas as pd
import numpy as np
import googleapiclient.discovery
from google.api_core.client_options import ClientOptions


def processing_data_from_excel(dataset_x):
    return_list = []
    for i in range(0, dataset_x.shape[0]):
        raw_value_from_excel = dataset_x[i,0]
        dataset_x[i,0] = raw_value_from_excel[1:]
        dataset_x[i,0] = dataset_x[i,0][:(len(dataset_x[i,0])-2)]
        current_numpy = np.fromstring(dataset_x[i,0], dtype=float, sep=' ')
        number_of_elements = current_numpy.shape[0]
        
        raw_value_from_excel = dataset_x[i,1]
        dataset_x[i,1] = raw_value_from_excel[1:]
        dataset_x[i,1] = dataset_x[i,1][:(len(dataset_x[i,1])-2)]
        voltage_numpy = np.fromstring(dataset_x[i,1], dtype=float, sep=' ')

        raw_value_from_excel = dataset_x[i,2]
        dataset_x[i,2] = raw_value_from_excel[1:]
        dataset_x[i,2] = dataset_x[i,2][:(len(dataset_x[i,2])-2)]
        charge_cap_numpy = np.fromstring(dataset_x[i,2], dtype=float, sep=' ')
        if current_numpy.shape[0] < 400 :
            dataset_x[i,0] = np.append(current_numpy, np.ones(400 - number_of_elements))
            
            dataset_x[i,1] = np.append(voltage_numpy, np.ones(400 - number_of_elements))
            dataset_x[i,2] = np.append(charge_cap_numpy, np.ones(400 - number_of_elements))
        else :
            dataset_x[i,0] = current_numpy[:400]
            
            dataset_x[i,1] = voltage_numpy[:400]
            dataset_x[i,2] = charge_cap_numpy[:400]
        x_model = [dataset_x[i,0], dataset_x[i,1], dataset_x[i,2]]
        x_model = np.array(x_model)
        x_model = np.transpose(x_model)
        return_list.append(x_model)

    
    return_list = np.array(return_list)
    print (return_list.shape)
    return return_list


def predict_json(project, region, model, instances, version=None):
    """Send json data to a deployed model for prediction.

    Args:
        project (str): project where the Cloud ML Engine Model is deployed.
        region (str): regional endpoint to use; set to None for ml.googleapis.com
        model (str): model name.
        instances ([Mapping[str: Any]]): Keys should be the names of Tensors
            your deployed model expects as inputs. Values should be datatypes
            convertible to Tensors, or (potentially nested) lists of datatypes
            convertible to tensors.
        version: str, version of the model to target.
    Returns:
        Mapping[str: any]: dictionary of prediction results defined by the
            model.
    """
    # Create the ML Engine service object.
    # To authenticate set the environment variable
    # GOOGLE_APPLICATION_CREDENTIALS=<path_to_service_account_file>
    prefix = "{}-ml".format(region) if region else "ml"
    api_endpoint = "https://{}.googleapis.com".format(prefix)
    client_options = ClientOptions(api_endpoint=api_endpoint)
    service = googleapiclient.discovery.build(
        'ml', 'v1', client_options=client_options)
    name = 'projects/{}/models/{}'.format(project, model)

    if version is not None:
        name += '/versions/{}'.format(version)

    response = service.projects().predict(
        name=name,
        body={"instances": instances}
    ).execute()

    if 'error' in response:
        raise RuntimeError(response['error'])

    return response['predictions']