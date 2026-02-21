from langchain_core.documents import Document

chunk = Document(
    page_content=('보험계약에서 정한 보험금의 지급사유가 발생한 경우 회사\n'
 '는 보험금을 지급하여 드립니다.- ① 제1항에서 지정한 특정질병의 합병증으로 인하여 진단\n'
 '- 확정된 특정질병 이외의 질병으로 계약에서 정한 보험\n'
 '- 금의 지급사유가 발생한 경우\n'
 '- ② 상해를 직접적인 원인으로 하여 보험금의 지급사유가\n'
 '- 발생한 경우\n'
 '- ③ 제1항에서 지정한 특정질병으로 인하여 사망하여 보험\n'
 '- 금의 지급사유가 발생한 경우\n'
 '\uf000 반려동물이 이 특별약관에서 정한 회사가 보험금을 지급\n'
 '하지 않는 기간의 종료일을 포함하여 계속하여 입원한 경우'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000549',
              'chunk_char_len': 285,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
