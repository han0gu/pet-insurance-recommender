from langchain_core.documents import Document

chunk = Document(
    page_content=('납처분절차에 따라 계약이 해지된 경우 해지 당시의 보험수익자가 계약자의 동의를\n'
 '얻어 계약 해지로 회사가 채권자에게 지급한 금액을 회사에 지급하고 제23조(특별약- 62 -# 관 내용의 변경 등) 제1항의 절차에 따라 '
 '계약자 명의를 보험수익자로 변경하여 특별\n'
 '약관의 특별부활(효력회복)을 청약할 수 있음을 보험수익자에게 통지하여야 합니다.<용어풀이># [강제집행과 담보권실행]강제집행이란 사법상 '
 '또는 행정법상의 의무를 이행하지 않는 사람에 대하여 국가가 강제 권력으로'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
