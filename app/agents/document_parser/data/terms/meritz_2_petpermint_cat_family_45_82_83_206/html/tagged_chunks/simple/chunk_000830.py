from langchain_core.documents import Document

chunk = Document(
    page_content=(". 기타 특별약관</p><p id='82' data-category='paragraph' style='font-size:20px'>1. "
 "반려동물 특정 질병 보장제한부 인수 특별약관</p><h1 id='83' style='font-size:18px'>제1조(특별약관의 체결 "
 "및 효력)</h1><br><p id='84' data-category='paragraph' "
 "style='font-size:16px'>\uf000 이 특별약관은 보험계약(특별약관이 부가된 경우에는 특<br>별약관을 포함합니다"),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000830',
              'chunk_char_len': 271,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
