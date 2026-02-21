from langchain_core.documents import Document

chunk = Document(
    page_content=('- ② 회사가 청약과 함께 제1회 보험료를 받고 청약을 승낙하기 전에 보험금 지급사유가\n'
 '- 발생하였을 때에도 보장개시일부터 이 약관이 정하는 바에 따라 보장을 합니다.\n'
 '<용어풀이># [보장개시일]회사가 보장을 개시하는 날로서 계약이 성립되고 제1회 보험료를 받은 날을 말하나, 회사가 승낙\n'
 '하기 전이라도 청약과 함께 제1회 보험료를 받은 경우에는 제1회 보험료를 받은 날을 말합니다.\n'
 '또한, 보장개시일을 계약일로 봅니다.③ 회사는 제2항에도 불구하고 다음 중 한 가지에 해당되는 경우에는 보장을 하지 않습'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000211',
              'chunk_char_len': 282,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
