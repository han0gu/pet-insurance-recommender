from langchain_core.documents import Document

chunk = Document(
    page_content=('약관을 말합니다.\n'
 '【재가입 적용대상 특별약관】\n'
 '･ 펫퍼민트 반려묘 입원의료비보장 특별약관 ･ 펫퍼민트 반려묘 통원의료비보장 특별약관 ･ 펫퍼민트 반려묘 입원의료비Ⅱ보장 특별약관 ･ '
 '펫퍼민트 반려묘 통원의료비Ⅱ보장 특별약관 ･ 펫퍼민트 반려묘 입원의료비Ⅲ보장 특별약관 ･ 펫퍼민트 반려묘 통원의료비Ⅲ보장 특별약관'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 160,
         'page': 98},
 'term_type': 'special',
 'clause': {'clause_type': 'renewal', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000247',
              'chunk_char_len': 173,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
