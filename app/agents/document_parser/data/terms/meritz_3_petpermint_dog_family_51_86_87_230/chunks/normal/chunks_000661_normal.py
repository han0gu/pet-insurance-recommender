from langchain_core.documents import Document

chunk = Document(
    page_content=('\uf000 이 특별약관에서 정한 회사가 보험금을 지급하지 않는 기간 중에 회사가 지정한 질병(이하「특정질병」이라 합니 '
 '다)(【별첨(특정질병 분류표(반려견))】)을 직접적인 원인 으로 계약에서 정한 보험금 지급사유가 발생한 경우에는 회 사는 보험금을 '
 '지급하지 않습니다'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 192},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000661',
              'chunk_char_len': 144,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
