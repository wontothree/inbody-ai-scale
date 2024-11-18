import uuid

from firebase import FireBase

class InBody:
    def __init__(self):
        self.inbody_data = {}

if __name__ == "__main__":
    firebase = FireBase()
    inbody = InBody()

    user_uuid = "134"
    height = 173
    weight = 70
    bmi = 1
    body_fat_percentage = 1
    body_fat_mass = 1

    inbody.inbody_data[user_uuid] = {
        "height": height,
        "weight": weight,
        "bmi": bmi,
        "body_fat_percentage": body_fat_percentage,
        "body_fat_mass": body_fat_mass
    }

    # upload data to firebase 
    firebase.upload_inbody_data(inbody.inbody_data)

