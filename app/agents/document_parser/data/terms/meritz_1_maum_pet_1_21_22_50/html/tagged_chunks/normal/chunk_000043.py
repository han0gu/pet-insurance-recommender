from langchain_core.documents import Document

chunk = Document(
    page_content=('. 귀 성형, 꼬리 성형, 성대제거 및 미용성형을 위한 처치에 따른 비용<br>11. 손톱절제(며느리발톱 제거 포함), 잔존유치, '
 '잠복고환, 제대허니아(배꼽부위탈장), 항<br>문낭 제거 등 건강동물에 실시하는 외과수술 및 기타 검사 또는 점안, 귀청소 '
 '등의<br>관리 비용<br>12. 서혜부허니아(서혜부탈장), 첩모난생(속눈썹 질환), 눈물샘으로 인한 비용<br>13'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive', 'eye']},
 'indexing': {'chunk_id': 'chunk_000043',
              'chunk_char_len': 207,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
