from langchain_core.documents import Document

chunk = Document(
    page_content=('【재가입 적용대상 특별약관】\n'
 '･ 펫퍼민트 반려견 입원의료비보장 특별약관 ･ 펫퍼민트 반려견 통원의료비보장 특별약관 ･ 펫퍼민트 반려견 입원의료비Ⅱ보장 특별약관 ･ '
 '펫퍼민트 반려견 통원의료비Ⅱ보장 특별약관 ･ 펫퍼민트 반려견 입원의료비Ⅲ보장 특별약관 ･ 펫퍼민트 반려견 통원의료비Ⅲ보장 특별약관'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 102},
 'term_type': 'special',
 'clause': {'clause_type': 'renewal', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000243',
              'chunk_char_len': 163,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
