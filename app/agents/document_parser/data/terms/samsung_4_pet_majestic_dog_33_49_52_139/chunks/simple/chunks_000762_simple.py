from langchain_core.documents import Document

chunk = Document(
    page_content=('보상합니다.\n'
 '3. 제3조(보상하는 손해) 제2항 제2호 다.목 또는 라.목의 비용 : 이 비용과 제1호에 의한 보상액의 합계액을 보상한도액 내에서 '
 '보상합니다.\n'
 '제8조 (손해방지의무)\n'
 '① 보험사고가 생긴 때에는 계약자 또는 피보험자는 아래의 사항을 이행하여야 합니다.'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 91,
         'page': 122},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000762',
              'chunk_char_len': 148,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
