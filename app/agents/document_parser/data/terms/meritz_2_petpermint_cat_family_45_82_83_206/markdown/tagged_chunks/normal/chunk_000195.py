from langchain_core.documents import Document

chunk = Document(
    page_content=('급합니다.# 제15조(재가입)\uf000 이 특별약관에서 재가입 적용대상 특별약관(이하「재가\n'
 '입 적용대상 특별약관」이라 합니다)이라 함은 아래의 특별97# 약관을 말합니다.# 【재가입 적용대상 특별약관】- ･ 펫퍼민트 반려묘 '
 '입원의료비보장 특별약관\n'
 '- ･ 펫퍼민트 반려묘 통원의료비보장 특별약관\n'
 '- ･ 펫퍼민트 반려묘 입원의료비Ⅱ보장 특별약관\n'
 '- ･ 펫퍼민트 반려묘 통원의료비Ⅱ보장 특별약관\n'
 '- ･ 펫퍼민트 반려묘 입원의료비Ⅲ보장 특별약관\n'
 '- ･ 펫퍼민트 반려묘 통원의료비Ⅲ보장 특별약관\n'
 '\uf000 재가입 적용대상 특별약관이 다음 각 호의 조건을 충족'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'renewal', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000195',
              'chunk_char_len': 299,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
