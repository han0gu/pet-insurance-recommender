from langchain_core.documents import Document

chunk = Document(
    page_content=('보험기간 종료 후 계약<br>자가 재가입을 원하는 경우 계약자는 재가입 시점에서 회사<br>가 판매하는 동일하거나 객관적이고 합리적인 '
 '범위내에서<br>기존 계약내용에 상응한 반려동물보험 상품(보험업감독규정<br>제1-2조(정의)에서 정한 장기손해보험에 한하며 '
 '이하「반려<br>동물보험 상품」이라 합니다)으로 가입을 할 수 있으며, 회<br>사는 이를 거절할 수 없습니다'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000365',
              'chunk_char_len': 206,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
