from langchain_core.documents import Document

chunk = Document(
    page_content=('제 6조 (보험금의 분담)\n'
 '① 이 특별약관에서 보장하는 위험과 같은 위험을 보장하는 다른 계약(공제계약을 포함합 니다)이 있을 경우 각 계약에 대하여 다른 계약이 '
 '없는 것으로 하여 각각 산출한 보상 책임액의 합계액이 손해액을 초과할 때에는 회사는 아래에 따라 손해를 보상합니다.\n'
 '② 피보험자가 다른 계약에 대하여 보험금 청구를 포기한 경우에도 회사의 제1항에 의한 지급보험금 결정에는 영향을 미치지 않습니다.\n'
 '제 7조 (특별약관의 소멸)'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 91,
         'page': 113},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000678',
              'chunk_char_len': 244,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
