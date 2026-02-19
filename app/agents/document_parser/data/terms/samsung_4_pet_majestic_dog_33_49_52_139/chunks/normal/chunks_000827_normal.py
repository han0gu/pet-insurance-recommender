from langchain_core.documents import Document

chunk = Document(
    page_content=('① 회사는 보험계약의 내용에도 불구하고 피보험자가 보험기간 중에 이륜자동차를 운전 (탑승을 포함합니다. 이하 같습니다)하는 중에 발생한 '
 '급격하고도 우연한 외래의 상해 사고를 직접적인 원인으로 보험계약에서 정한 보험금 지급사유가 발생한 경우에 보험 금을 지급하지 않습니다. '
 '다만, 피보험자가 이륜자동차를 직업, 직무 또는 동호회 활 동과 출퇴근용도 등 주로 사용하게 된 사실을 회사가 입증하지 못한 때에는 '
 '보험금을 지급합니다'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 91,
         'page': 132},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000827',
              'chunk_char_len': 236,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
