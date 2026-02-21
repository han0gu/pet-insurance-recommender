from langchain_core.documents import Document

chunk = Document(
    page_content=('- 험료(정산금액을 포함합니다)를 계약자가 납입하지 않았을 때, 회사는 위험이 증가되\n'
 '- 기 전에 적용된 보험요율(이하「변경전 요율」이라 합니다)의 위험이 증가된 후에 적\n'
 '- 용해야 할 보험요율(이하「변경후 요율」이라 합니다)에 대한 비율에 따라 보험금을\n'
 '- 삭감하여 지급합니다. 다만, 증가된 위험과 관계없이 발생한 보험금 지급사유에 관해\n'
 '- 서는 원래대로 지급합니다.\n'
 '<예시안내># [비례 보상]보험기간 중 직업의 변경으로 위험이 증가(상해급수 1급 → 2급)되었으나, 이를 회사에 알리지'),
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
