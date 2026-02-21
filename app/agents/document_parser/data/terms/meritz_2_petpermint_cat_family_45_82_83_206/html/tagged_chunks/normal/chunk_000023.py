from langchain_core.documents import Document

chunk = Document(
    page_content=('. 다만,【별표<br>2(장해분류표)】에 장해판정시기를 별도로 정한 경우에는<br>그에 따릅니다.<br>\uf000 제1항에 따라 '
 '장해지급률이 결정되었으나 그 이후 보장<br>을 받을 수 있는 기간(보장의 효력이 없어진 경우에는 보험<br>기간이 10년 이상인 보장은 '
 '상해 발생일부터 2년 이내로 하<br>고, 보험기간이 10년 미만인 보장은 상해 발생일부터 1년이<br>내)에 장해상태가 더 악화된 '
 '때에는 그 악화된 장해상태를<br>기준으로 장해지급률을 결정합니다.<br>\uf000【별표2(장해분류표)】에 해당되지 않는 후유장해는 '
 '피보<br>험자의'),
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
 'indexing': {'chunk_id': 'chunk_000023',
              'chunk_char_len': 298,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
