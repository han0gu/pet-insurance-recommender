from langchain_core.documents import Document

chunk = Document(
    page_content=('- 2. 제2조(특별면책조건의 내용) 제1항 제2호의 특정질병에 대하여 면책을 조건으로\n'
 '- 체결한 후 보장개시일(책임개시일) 이전에 동일한 특정질병이 발생한 경우\n'
 '# 제2조 (특별면책조건의 내용)① 이 특별약관에서 정한 회사가 보험금을 지급하지 않는 기간 중에 다음 각 호의 질병을\n'
 '직접적인 원인으로 보험계약에서 정한 보험금 지급사유가 발생한 경우에 회사는 보험금을 지급하지 않으며, 보험료 납입면제사유 및 유사암 '
 '납입지원 사유가 발생한 경우\n'
 '에 회사는 보험료 납입을 면제 또는 지원하지 않습니다. 다만, 질병으로 인한 사망 또'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000724',
              'chunk_char_len': 295,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
