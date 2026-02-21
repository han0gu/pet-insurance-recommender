from langchain_core.documents import Document

chunk = Document(
    page_content=('- ③ 회사는 보험금을 지급하지 않는 사유 등 계약자나 피보험자에게 불리하거나 부담을\n'
 '- 주는 내용은 확대하여 해석하지 않습니다.\n'
 '# 제43조 (설명서 교부 및 보험안내자료 등의 효력)- ① 회사는 일반금융소비자에게 청약을 권유하거나 일반금융소비자가 설명을 요청하는 '
 '경\n'
 '- 우 보험상품에 관한 중요한 사항을 계약자가 이해할 수 있도록 설명하고 계약자가 이\n'
 '- 해하였음을 서명(「전자서명법」 제2조 제2호에 따른 전자서명을 포함), 기명날인 또\n'
 '- 는 녹취 등을 통해 확인받아야 하며, 설명서를 제공하여야 합니다.'),
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
