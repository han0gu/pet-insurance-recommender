from langchain_core.documents import Document

chunk = Document(
    page_content=('② 「호스피스·완화의료 및 임종과정에 있는 환자의 연명의료결정에 관한 법률」에 따 른 연명의료중단 등 결정 및 그 이행으로 피보험자가 '
 '사망하는 경우 연명의료중단 등 결정 및 그 이행은 제1조(보험금의 지급사유) ‘사망’의 원인 및 ‘사망보험금’ 지 급에 영향을 미치지 '
 '않습니다.'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 91,
         'page': 88},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000473',
              'chunk_char_len': 155,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
