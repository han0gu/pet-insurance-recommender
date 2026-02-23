from langchain_core.documents import Document

chunk = Document(
    page_content=('는 바에 따라 그 지급을 보장합니다.# <용어풀이># [예금자보호제도]예금자보호제도란 예금보험공사에서 금융기관 등으로부터 미리 보험료를 '
 '받아 적립해 두었다가 금\n'
 '융기관이 경영악화나 파산 등으로 예금을 지급할 수 없는 경우 해당 금융기관을 대신하여 예금자\n'
 '에게 보험금 또는 환급금을 지급함으로써 예금자를 보호하는 제도를 말합니다. 본 회사에 있는 모'),
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
