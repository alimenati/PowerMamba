# Data Scraper for ERCOT Wind Generation Data

This script, `datascrapper.py`, provides an example of how to download hourly wind generation data from the ERCOT website. 

## Requirements

To use the script, you'll need the following credentials:
- **Username**
- **Password**
- **Access Token**
- **Subscription Key**

These credentials can be obtained from ERCOT's API platform. For more information, refer to the [ERCOT API Documentation](https://apiexplorer.ercot.com/support).

## Customization for Other Datasets

The script includes a function to fetch wind power data:

```python
def get_wind_power_data(access_token, page, datefrom, dateto):
    url = "https://api.ercot.com/api/public-reports/archive/DATASET_NAME"  # Enter the Dataset ID from ERCOT website
```
