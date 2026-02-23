from langchain_core.documents import Document

chunk = Document(
    page_content=('1,000만원에 대한 대위권만 가지며 피보험자는 제3자에 대해 1,000만원을 제외한 나머지 손해금\n'
 '액에 대한 손해배상청구권을 가집니다.- ② 계약자 또는 피보험자는 제1항에 의하여 회사가 취득한 권리를 행사하거나 지키는 것\n'
 '- 에 관하여 필요한 조치를 하여야 하며, 또한 회사가 요구하는 증거 및 서류를 제출하\n'
 '- 여야 합니다. 이에 필요한 비용은 회사가 드립니다.\n'
 '- ③ 회사는 제1항 및 제2항에도 불구하고 타인을 위한 계약의 경우에는 계약자에 대한 대\n'
 '- 위권을 포기합니다.'),
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
