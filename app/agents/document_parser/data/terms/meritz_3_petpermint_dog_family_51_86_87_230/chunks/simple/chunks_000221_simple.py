from langchain_core.documents import Document

chunk = Document(
    page_content=('사실대로 알리지 않게 하였거나 부실한 사항을 알릴 것을 권유했을 때. 다만, 보험설계사 등의 행위가 없 었다 하더라도 계약자 또는 '
 '피보험자가 사실대로 알리 지 않거나 부실한 사항을 알렸다고 인정되는 경우에는 계약을 해지할 수 있습니다.'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 98},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000221',
              'chunk_char_len': 131,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
