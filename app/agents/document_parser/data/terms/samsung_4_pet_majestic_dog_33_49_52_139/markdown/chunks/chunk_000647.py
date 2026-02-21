from langchain_core.documents import Document

chunk = Document(
    page_content=('- 으로 봅니다.\n'
 '# 제6조 (손해의 통지 및 조사)① 계약자 또는 피보험자는 아래와 같은 사실이 있는 경우에는 지체없이 그 내용을 회사\n'
 '에 알려야 합니다.- 1. 사고가 발생하였을 경우 사고가 발생한 때와 곳, 피해자의 주소와 성명, 사고상황\n'
 '- 및 이들 사항의 증인이 있을 경우 그 주소와 성명\n'
 '- 2. 피해자로부터 손해배상청구를 받았을 경우\n'
 '- 3. 피해자로부터 손해배상책임에 관한 소송을 제기받았을 경우\n'
 '② 계약자 또는 피보험자가 제1항 각 호의 통지를 게을리하여 손해가 증가된 때에는 회'),
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
