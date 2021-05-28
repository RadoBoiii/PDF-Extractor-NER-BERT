import numpy as np

import joblib
import torch

import config
import dataset
import engine
from model import EntityModel


if __name__ == "__main__":

    meta_data = joblib.load(config.META_FILE)
    enc_pos = meta_data["enc_pos"]
    enc_tag = meta_data["enc_tag"]

    num_pos = len(list(enc_pos.classes_))
    num_tag = len(list(enc_tag.classes_))

    sentence = """
    Opportunity Medical Center Account # Admit Date Admit Time Reg Init Brought By Info Provided By MR Number 02487 02/24/202 0o.00 EG Friend Wife 54391500 Admitting Physician Primary Care Phys Room # Type Service Discharge Date Christopher Johnson David White 913 Surgery Patient Last Name First Middle Former Name Race Rel Pref Social Security # Robertson Jeffrey yes 066-58-6681 Patient Address Apt: No_ State Zip Code Patient Phone # 5402 Parker Radial Suite 186 New Hanna  KY 42363 446.952.9017 Driver's License # Age Birth Date Birthplace Gender MS Occupation Accident? DatelTime 20 01/07/201/ KY Geographici] Patient Employer Employer Address Employer Phone Jenkins and Sons 4303 Waters Loop 916-791-2158 Spouse Name Spouse Address State Spouse Phone Joshua Moore 924 Carol Pike North Christoll IN +1-872-239-157 Emergency Contact Relationship Home Phone Cell Phone Work Caitlin Marshall Mother 3348301877 3348309409 Admitting Diagnosis Admit Type ICD9 Admit Source Prostate disease Surgery Primary Insurance Plan Primary Policy # Authorization # Primary Policy Holder Northern Care 74-7790-3008 Insurance Plan #2 Secondary Authorization # Secondary Policy Holder Eastern Care 945-3941 Insurance Plan #3 Tertiary Policy # Authorization # Tertiary Policy Holder Guarantor Name Rel to Pt Mailing Address Guarantor Phone Guarantor Occupation Employer Employer Address Employer Phone Billing Remarks: Principal Diagnosis: Prostate disease Code: 41313 Secondary Diagnosis: Code: Operations and Procedures: Physician Date Code Consulting Physician: Final Disposition: Discharged Transferred Left AMA Expired Autopsy Yes No certify that my identification of the principal secondary diagnosis and the procedures performed is accurate to the best of my knowledge_ Opportunity Medical Center Attending Physician Date Time City City Phone Policy and 
    """
    tokenized_sentence = config.TOKENIZER.encode(sentence)

    sentence = sentence.split()
    print(sentence)
    print(tokenized_sentence)

    test_dataset = dataset.EntityDataset(
        texts=[sentence], 
        pos=[[0] * len(sentence)], 
        tags=[[0] * len(sentence)]
    )

    device = torch.device("cuda")
    model = EntityModel(num_tag=num_tag, num_pos=num_pos)
    model.load_state_dict(torch.load(config.MODEL_PATH))
    model.to(device)

    with torch.no_grad():
        data = test_dataset[0]
        for k, v in data.items():
            data[k] = v.to(device).unsqueeze(0)
        tag, pos, _ = model(**data)

        print(
            enc_tag.inverse_transform(
                tag.argmax(2).cpu().numpy().reshape(-1)
            )[:len(tokenized_sentence)]
        )
        print(
            enc_pos.inverse_transform(
                pos.argmax(2).cpu().numpy().reshape(-1)
            )[:len(tokenized_sentence)]
        )
