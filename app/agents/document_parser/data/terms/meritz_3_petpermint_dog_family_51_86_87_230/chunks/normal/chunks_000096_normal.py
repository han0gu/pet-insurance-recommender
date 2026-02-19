from langchain_core.documents import Document

chunk = Document(
    page_content=('출이율」이라 하며, 회사에서 별도로 정한 방법에 따라 결정합니다. 보험계약대출은 순수보장성 상품 등 보험상 품의 종류 및 보험계약 '
 '경과기간에 따라 제한 될 수 있 습니다.\n'
 '제22조(계약의 무효)'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 71},
 'term_type': 'basic',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000096',
              'chunk_char_len': 108,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
