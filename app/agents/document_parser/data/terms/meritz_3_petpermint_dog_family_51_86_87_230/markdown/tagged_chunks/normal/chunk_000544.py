from langchain_core.documents import Document

chunk = Document(
    page_content=('험료 및 회사의 보장개시)에서 정한 보장개시일과 동일합니\n'
 '다.\n'
 '\uf000 계약이 해지, 기타사유에 따라 효력이 없는 경우에는 이\n'
 '특별약관도 더 이상 효력이 없습니다.\n'
 '\uf000 이 특별약관에서 정한 보장개시일 이전에 발생한 질병에\n'
 '대하여 계약을 무효로 하는 경우에도 제2조(특별면책조건의\n'
 '내용) 제1항에서 정한 특정질병에 대하여 면책을 조건으로\n'
 '체결한 후 보장개시일 이전에 동일한 특정질병이 발생한 경\n'
 '우에는 계약을 무효로 하지 않습니다.제2조(특별면책조건의 내용)\uf000 이 특별약관에서 정한 회사가 보험금을 지급하지 않는'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000544',
              'chunk_char_len': 282,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
