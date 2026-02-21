from langchain_core.documents import Document

chunk = Document(
    page_content=('# 제5조(보험금의 지급절차)\uf000 회사는 제4조(보험금의 청구)에서 정한 서류를 접수한\n'
 '때에는 접수증을 드리고, 그 서류를 접수받은 후 지체없이\n'
 '지급할 보험금을 결정하고 지급할 보험금이 결정되면 7일\n'
 '이내에 이를 지급합니다. 그러나 지급할 보험금이 결정되기\n'
 '전이라도 피보험자의 청구가 있을 때에는 회사가 추정한 보\n'
 '험금의 50% 상당액을 가지급보험금으로 지급합니다.\uf000 회사는 제1항의 지급보험금이 결정된 후 7일(이하「지급\n'
 '기일」이라 합니다)이 지나도록 보험금을 지급하지 않았을'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000490',
              'chunk_char_len': 266,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
