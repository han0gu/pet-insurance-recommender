from langchain_core.documents import Document

chunk = Document(
    page_content=('는 요금을 말하며, 보험료는「보장보험료」와「적립보험\n'
 '료」로 구성되어 있습니다.\n'
 '또한, 보험료는 보험금 지급을 위한 위험보험료, 회사가\n'
 '적립한 금액을 돌려주기 위한 적립부분 순보험료 및 회\n'
 '사의 사업경비를 위한 부가보험료로 구성됩니다.49# 보험료 = 보장보험료 + 적립보험료\n'
 '보장보험료 = 위험보험료 + 부가보험료\n'
 '적립보험료 = 적립부분 순보험료 + 부가보험료제2관 보험금의 지급제3조(보험금의 지급사유)회사는 보험증권에 기재된 피보험자가 보험기간 '
 '중에 상해\n'
 '로【별표2(장해분류표)】에서 정한 장해지급률이 80%이상에'),
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
 'indexing': {'chunk_id': 'chunk_000011',
              'chunk_char_len': 289,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
