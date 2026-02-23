from langchain_core.documents import Document

chunk = Document(
    page_content=('- 관)에서 규정한 국내의 병원 또는 의원에서 의사의 관리 하에 [별표-상해및질병관련\n'
 '- 1]급여 창상봉합술 대상 수가코드에서 정한 진료행위로 치료를 받은 경우를 말합니\n'
 '- 다.\n'
 '- ③ 이 특별약관에서 「안면부 창상봉합술(급여)」 이라 함은 병원 또는 의원의 의사에 의하\n'
 '- 여 치료가 필요하다고 인정된 경우로서 자택 등에서의 치료가 곤란하여 의료법 제3조\n'
 '- (의료기관)에서 규정한 국내의 병원 또는 의원에서 의사의 관리 하에 [별표-상해및질\n'
 '- 병관련2]급여 창상봉합술(안면부) 대상 수가코드에서 정한 진료행위로 치료를 받은'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000436',
              'chunk_char_len': 295,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
