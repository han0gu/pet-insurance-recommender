from langchain_core.documents import Document

chunk = Document(
    page_content=('잠복고환 | 고환이 음낭까지 내려오지 못하는 증상\n'
 '③ 제2항에서 정한 조치에 다른 진료를 병행하여 실시한 경우에는 제2항에서 정한 조치( 마취 비용을 포함합니다)에 대한 보험금은 지급하지 '
 '않습니다.\n'
 '제5조 (보험금의 청구)\n'
 '① 피보험자는 다음의 서류를 제출하고 보험금을 청구하여야 합니다.\n'
 '1. 보험금 청구서(회사 양식) 2. 등록묘의 경우에는 동물등록증 또는 등록번호 3. 미등록묘의 경우에는 가입동물의 사진 2매(얼굴전면, '
 '측면전신사진)를 회사에 제출\n'
 '하고 가입동물이 보험에 가입한 동물과 동일함을 확인 후 보험금을 지급합니다.'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 115},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000723',
              'chunk_char_len': 297,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
