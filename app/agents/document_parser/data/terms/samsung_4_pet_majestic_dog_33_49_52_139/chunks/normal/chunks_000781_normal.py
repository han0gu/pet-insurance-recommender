from langchain_core.documents import Document

chunk = Document(
    page_content=('제2조 (입원의 정의와 장소)\n'
 '이 특별약관에서 「입원」 이라 함은 병원 또는 의원의 의사, 치과의사 또는 한의사의 면허 를 가진 자(이하 「의사」 라 합니다)에 의하여 '
 '상해의 치료가 필요하다고 인정된 경우로서 자택 등에서 치료가 곤란하여 의료법 제3조(의료기관)에서 규정한 국내의 병원, 의원 또 는 '
 '국외의 의료관련법에서 정한 의료기관에 입실하여 의사의 관리하에 치료에 전념하는 것을 말합니다.\n'
 '제3조(보험금의 청구)\n'
 '① 피보험자가 보험금을 청구할 때에는 다음의 서류를 회사에 제출하여야 합니다.'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 91,
         'page': 124},
 'term_type': 'special',
 'clause': {'clause_type': 'definition', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000781',
              'chunk_char_len': 277,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
