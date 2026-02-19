from langchain_core.documents import Document

chunk = Document(
    page_content=('Chart Type: bar\n'
 '보장개시일 | 슬관절탈구\n'
 'item_01 | 4.1 | 2.6\n'
 'item_02 | 0.4 | 4.1\n'
 '※ 설명 보장개시일로부터 1년이내 발생한 슬관절탈구 : 보험금 미지급'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 156},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive', 'joint']},
 'indexing': {'chunk_id': 'chunk_000503',
              'chunk_char_len': 108,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
