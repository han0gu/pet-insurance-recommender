from langchain_core.documents import Document

chunk = Document(
    page_content=('경우 특정된 보험수익 자를 포함합니다)가 납입최고(독촉)기간 안에 보험료를 납 입하지 않은 경우에는 납입최고(독촉)기간이 끝나는 날의 '
 '다음날에 이 보장계약을 해제합니다'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 190},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000652',
              'chunk_char_len': 93,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
