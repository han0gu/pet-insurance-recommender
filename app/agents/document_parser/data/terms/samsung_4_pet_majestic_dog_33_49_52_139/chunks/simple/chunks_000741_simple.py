from langchain_core.documents import Document

chunk = Document(
    page_content=('① 계약관계 관련 용어\n'
 '1. 피보험자 : 보험사고로 인하여 타인에 대한 법률상 손해배상책임을 부담하는 손해 를 입은 사람(법인인 경우에는 그 이사 또는 법인의 '
 '업무를 집행하는 그 밖의 기관 )을 말합니다.\n'
 '② 보상 관련 용어'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 91,
         'page': 120},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000741',
              'chunk_char_len': 125,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
