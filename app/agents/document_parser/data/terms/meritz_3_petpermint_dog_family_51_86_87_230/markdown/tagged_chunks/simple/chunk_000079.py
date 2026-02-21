from langchain_core.documents import Document

chunk = Document(
    page_content=('결정합니다. 보험계약대출은 순수보장성 상품 등 보험상\n'
 '품의 종류 및 보험계약 경과기간에 따라 제한 될 수 있\n'
 '습니다.# 제22조(계약의 무효)\uf000 다음 중 한 가지에 해당하는 경우에는 계약을 무효로 하\n'
 '며 이미 납입한 보험료를 돌려드립니다. 다만, 회사의 고의\n'
 '또는 과실로 계약이 무효로 된 경우와 회사가 승낙 전에 무\n'
 '효임을 알았거나 알 수 있었음에도 보험료를 반환하지 않은\n'
 '경우에는 보험료를 납입한 날의 다음날부터 반환일까지의\n'
 '기간에 대하여 회사는 보험계약대출이율을 연단위 복리로'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000079',
              'chunk_char_len': 269,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
